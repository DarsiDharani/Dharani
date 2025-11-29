/**
 * Notification Service
 * 
 * Purpose: Service for managing user notifications
 * Features:
 * - Fetch notifications for the current user
 * - Get unread notification count
 * - Mark notifications as read
 * - Delete notifications
 * - Real-time notification updates
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject, interval } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { of } from 'rxjs';
import { ApiService } from './api.service';
import { AuthService } from './auth.service';
import { Notification, UnreadCountResponse } from '../models/notification.model';

@Injectable({
  providedIn: 'root'
})
export class NotificationService {
  private notificationsSubject = new BehaviorSubject<Notification[]>([]);
  public notifications$ = this.notificationsSubject.asObservable();

  private unreadCountSubject = new BehaviorSubject<number>(0);
  public unreadCount$ = this.unreadCountSubject.asObservable();

  private refreshInterval?: any;

  constructor(
    private http: HttpClient,
    private apiService: ApiService,
    private authService: AuthService
  ) {
    // Don't start polling in constructor - wait for explicit initialize() call
    // This ensures proper cleanup and prevents duplicate polling
  }

  /**
   * Get HTTP headers with authentication token
   */
  private getHeaders(): HttpHeaders {
    const token = this.authService.getToken();
    return new HttpHeaders({
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    });
  }

  /**
   * Fetch all notifications for the current user
   * @param unreadOnly - If true, only fetch unread notifications
   * @param limit - Maximum number of notifications to fetch
   */
  getNotifications(unreadOnly: boolean = false, limit: number = 50): Observable<Notification[]> {
    const params: any = { limit };
    if (unreadOnly) {
      params.unread_only = true;
    }

    return this.http.get<Notification[]>(
      this.apiService.notificationsUrl,
      {
        headers: this.getHeaders(),
        params
      }
    ).pipe(
      map(notifications => {
        // Sort by created_at descending (newest first) - mobile app behavior
        const sortedNotifications = [...notifications].sort((a, b) => {
          const dateA = new Date(a.created_at).getTime();
          const dateB = new Date(b.created_at).getTime();
          return dateB - dateA;
        });
        
        // Update the subject with sorted notifications
        this.notificationsSubject.next(sortedNotifications);
        return sortedNotifications;
      }),
      catchError(error => {
        console.error('Error fetching notifications:', error);
        // Return empty array but don't clear existing notifications on error
        // This prevents flickering when there's a temporary network issue
        return of(this.notificationsSubject.value);
      })
    );
  }

  /**
   * Get unread notification count
   */
  getUnreadCount(): Observable<number> {
    return this.http.get<UnreadCountResponse>(
      this.apiService.unreadCountUrl,
      { headers: this.getHeaders() }
    ).pipe(
      map(response => {
        this.unreadCountSubject.next(response.unread_count);
        return response.unread_count;
      }),
      catchError(error => {
        console.error('Error fetching unread count:', error);
        return of(0);
      })
    );
  }

  /**
   * Mark a notification as read
   * @param notificationId - ID of the notification to mark as read
   */
  markAsRead(notificationId: number): Observable<Notification> {
    return this.http.patch<Notification>(
      this.apiService.markNotificationReadUrl(notificationId),
      {},
      { headers: this.getHeaders() }
    ).pipe(
      map(notification => {
        // Update the notification in the list
        const currentNotifications = this.notificationsSubject.value;
        const updatedNotifications = currentNotifications.map(n =>
          n.id === notificationId ? notification : n
        );
        this.notificationsSubject.next(updatedNotifications);
        
        // Update unread count
        this.getUnreadCount().subscribe();
        
        return notification;
      }),
      catchError(error => {
        console.error('Error marking notification as read:', error);
        throw error;
      })
    );
  }

  /**
   * Mark all notifications as read
   */
  markAllAsRead(): Observable<any> {
    return this.http.patch(
      this.apiService.markAllNotificationsReadUrl,
      {},
      { headers: this.getHeaders() }
    ).pipe(
      map(() => {
        // Update all notifications to read
        const currentNotifications = this.notificationsSubject.value;
        const updatedNotifications = currentNotifications.map(n => ({
          ...n,
          is_read: true,
          read_at: new Date().toISOString()
        }));
        this.notificationsSubject.next(updatedNotifications);
        
        // Update unread count
        this.unreadCountSubject.next(0);
        
        return { success: true };
      }),
      catchError(error => {
        console.error('Error marking all notifications as read:', error);
        throw error;
      })
    );
  }

  /**
   * Delete a notification
   * @param notificationId - ID of the notification to delete
   */
  deleteNotification(notificationId: number): Observable<any> {
    return this.http.delete(
      this.apiService.deleteNotificationUrl(notificationId),
      { headers: this.getHeaders() }
    ).pipe(
      map(() => {
        // Remove notification from list
        const currentNotifications = this.notificationsSubject.value;
        const filteredNotifications = currentNotifications.filter(n => n.id !== notificationId);
        this.notificationsSubject.next(filteredNotifications);
        
        // Update unread count
        this.getUnreadCount().subscribe();
        
        return { success: true };
      }),
      catchError(error => {
        console.error('Error deleting notification:', error);
        throw error;
      })
    );
  }

  /**
   * Start polling for new notifications (every 10 seconds for mobile-like behavior)
   */
  startPolling(): void {
    // Clear any existing interval
    this.stopPolling();
    
    // Don't fetch here - initialize() will call refreshNotifications first
    // This prevents duplicate calls
    
    // Poll every 10 seconds (mobile apps typically poll every 5-15 seconds)
    this.refreshInterval = setInterval(() => {
      this.refreshNotifications();
    }, 10000);
  }

  /**
   * Stop polling for notifications
   */
  stopPolling(): void {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = undefined;
    }
  }

  /**
   * Refresh notifications and unread count
   * This is called automatically by polling and can be called manually
   */
  refreshNotifications(): void {
    if (!this.authService.isLoggedIn()) {
      // User is not logged in, stop polling and clear
      this.stopPolling();
      this.notificationsSubject.next([]);
      this.unreadCountSubject.next(0);
      return;
    }
    
    // Fetch notifications and unread count in parallel
    this.getNotifications().subscribe({
      next: () => {
        // Notifications updated via BehaviorSubject in getNotifications()
      },
      error: (error) => {
        console.error('Error refreshing notifications:', error);
        // Don't clear notifications on error - keep existing ones
      }
    });
    
    this.getUnreadCount().subscribe({
      next: () => {
        // Unread count updated via BehaviorSubject in getUnreadCount()
      },
      error: (error) => {
        console.error('Error refreshing unread count:', error);
      }
    });
  }
  
  /**
   * Force immediate refresh (useful for manual refresh)
   */
  forceRefresh(): void {
    this.refreshNotifications();
  }

  /**
   * Initialize notifications when user logs in
   * This should be called once when user logs in or component initializes
   * Mobile app behavior: Shows existing notifications immediately, then refreshes
   */
  initialize(): void {
    // Only initialize if user is logged in
    if (!this.authService.isLoggedIn()) {
      return;
    }
    
    // Stop any existing polling first
    this.stopPolling();
    
    // Don't clear notifications - let them persist (mobile app behavior)
    // They will be refreshed/updated with fresh data from server
    
    // Fetch initial notifications immediately (fast!)
    this.refreshNotifications();
    
    // Start polling for updates
    this.startPolling();
    
    // Set up visibility change listener (refresh when user comes back to tab)
    this.setupVisibilityListener();
  }
  
  /**
   * Setup visibility API listener to refresh when app comes back into focus
   * This mimics mobile app behavior when user switches back to the app
   */
  private setupVisibilityListener(): void {
    // Remove existing listener if any
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    
    // Add new listener
    document.addEventListener('visibilitychange', this.handleVisibilityChange);
  }
  
  /**
   * Handle visibility change - refresh notifications when user returns to tab
   */
  private handleVisibilityChange = (): void => {
    if (document.visibilityState === 'visible' && this.authService.isLoggedIn()) {
      // User returned to the tab - refresh immediately
      this.refreshNotifications();
    }
  }

  /**
   * Clear notifications when user logs out
   * Only stops polling - doesn't clear notifications (mobile app behavior)
   * Notifications will be refreshed on next login
   */
  clear(): void {
    this.stopPolling();
    // Don't clear notifications - they'll be refreshed on next login
    // This mimics mobile app behavior where notifications persist
    
    // Remove visibility listener
    document.removeEventListener('visibilitychange', this.handleVisibilityChange);
  }
  
  /**
   * Delete all notifications (clear all)
   */
  deleteAllNotifications(): Observable<any> {
    const url = this.apiService.deleteAllNotificationsUrl;
    const headers = this.getHeaders();
    
    console.log('Deleting all notifications - URL:', url);
    console.log('Headers:', headers);
    
    // Temporarily stop polling to prevent refresh during deletion
    const wasPolling = !!this.refreshInterval;
    if (wasPolling) {
      this.stopPolling();
    }
    
    return this.http.delete(url, { headers }).pipe(
      map((response: any) => {
        console.log('Delete all response received:', response);
        
        // Clear all notifications from local state IMMEDIATELY
        // Don't refresh - we just deleted everything, so we know it's empty
        this.notificationsSubject.next([]);
        this.unreadCountSubject.next(0);
        
        console.log('All notifications deleted successfully, local state cleared');
        
        // Restart polling after a short delay to ensure deletion is complete
        if (wasPolling) {
          setTimeout(() => {
            this.startPolling();
          }, 2000); // Wait 2 seconds before restarting polling
        }
        
        return { success: true, ...response };
      }),
      catchError(error => {
        console.error('Error deleting all notifications:', error);
        console.error('Error details:', {
          status: error.status,
          statusText: error.statusText,
          error: error.error,
          message: error.message
        });
        
        // Restart polling on error
        if (wasPolling) {
          this.startPolling();
        }
        
        // On error, don't clear local state - keep existing notifications
        throw error;
      })
    );
  }
}

