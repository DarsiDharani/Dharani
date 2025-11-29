/**
 * Notifications Component
 * 
 * Purpose: Display user notifications in a mobile-app-like interface
 * Features:
 * - Slide-in notifications from the top
 * - Notification bell icon with unread count badge
 * - Notification dropdown panel
 * - Mark as read functionality
 * - Click to navigate to related content
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit, OnDestroy, ViewChild, ElementRef, HostListener } from '@angular/core';
import { Router } from '@angular/router';
import { Subscription } from 'rxjs';
import { NotificationService } from '../../services/notification.service';
import { Notification } from '../../models/notification.model';
import { trigger, state, style, transition, animate, query, stagger } from '@angular/animations';

@Component({
  selector: 'app-notifications',
  templateUrl: './notifications.component.html',
  styleUrls: ['./notifications.component.css'],
  animations: [
    trigger('slideIn', [
      transition(':enter', [
        style({ transform: 'translateY(-100%)', opacity: 0 }),
        animate('300ms ease-out', style({ transform: 'translateY(0)', opacity: 1 }))
      ]),
      transition(':leave', [
        animate('300ms ease-in', style({ transform: 'translateY(-100%)', opacity: 0 }))
      ])
    ]),
    trigger('slideDown', [
      transition(':enter', [
        style({ transform: 'translateY(-10px)', opacity: 0 }),
        animate('200ms ease-out', style({ transform: 'translateY(0)', opacity: 1 }))
      ]),
      transition(':leave', [
        animate('150ms ease-in', style({ transform: 'translateY(-10px)', opacity: 0 }))
      ])
    ]),
    trigger('listAnimation', [
      transition('* => *', [
        query(':enter', [
          style({ opacity: 0, transform: 'translateX(-20px)' }),
          stagger(50, [
            animate('200ms ease-out', style({ opacity: 1, transform: 'translateX(0)' }))
          ])
        ], { optional: true })
      ])
    ]),
    trigger('pulse', [
      state('active', style({ transform: 'scale(1.1)' })),
      state('inactive', style({ transform: 'scale(1)' })),
      transition('inactive => active', animate('200ms ease-in')),
      transition('active => inactive', animate('200ms ease-out'))
    ]),
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('200ms ease-out', style({ opacity: 1 }))
      ]),
      transition(':leave', [
        animate('150ms ease-in', style({ opacity: 0 }))
      ])
    ]),
    trigger('modalScale', [
      transition(':enter', [
        style({ opacity: 0, transform: 'scale(0.95)' }),
        animate('200ms ease-out', style({ opacity: 1, transform: 'scale(1)' }))
      ]),
      transition(':leave', [
        animate('150ms ease-in', style({ opacity: 0, transform: 'scale(0.95)' }))
      ])
    ])
  ]
})
export class NotificationsComponent implements OnInit, OnDestroy {
  @ViewChild('notificationPanel') notificationPanel?: ElementRef;
  @ViewChild('bellIcon') bellIcon?: ElementRef;

  notifications: Notification[] = [];
  unreadCount: number = 0;
  isPanelOpen: boolean = false;
  showNewNotifications: boolean = false;
  newNotifications: Notification[] = [];
  isRefreshing: boolean = false;
  isClearing: boolean = false;
  showClearConfirmModal: boolean = false;

  private notificationsSubscription?: Subscription;
  private unreadCountSubscription?: Subscription;

  constructor(
    private notificationService: NotificationService,
    private router: Router
  ) {}

  ngOnInit(): void {
    // Initialize notification service (start polling)
    // This will fetch notifications and start polling
    this.notificationService.initialize();
    
    // Subscribe to notifications
    this.notificationsSubscription = this.notificationService.notifications$.subscribe(
      notifications => {
        // If we're currently clearing, ignore updates from polling
        // This prevents polling from restoring notifications during deletion
        if (this.isClearing) {
          console.log('Ignoring notification update during clear operation');
          return;
        }
        
        // Check for new notifications (compare with previous list)
        const previousIds = this.notifications.map(n => n.id);
        const newOnes = notifications.filter(n => !previousIds.includes(n.id) && !n.is_read);
        
        // Show new notifications as slide-in from top (mobile-like behavior)
        // Show even on first load if there are unread notifications
        if (newOnes.length > 0) {
          // Only show slide-in if we already have some notifications loaded
          // OR if these are the first notifications and they're unread
          if (this.notifications.length > 0 || (notifications.length > 0 && newOnes.length > 0)) {
            this.newNotifications = newOnes;
            this.showNewNotifications = true;
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
              this.showNewNotifications = false;
              setTimeout(() => {
                this.newNotifications = [];
              }, 300);
            }, 5000);
          }
        }
        
        // Update notifications list
        this.notifications = notifications;
      }
    );

    // Subscribe to unread count
    this.unreadCountSubscription = this.notificationService.unreadCount$.subscribe(
      count => {
        this.unreadCount = count;
      }
    );
    
    // Note: Don't call loadNotifications() here because initialize() already does it
    // This prevents duplicate API calls
  }

  ngOnDestroy(): void {
    this.notificationsSubscription?.unsubscribe();
    this.unreadCountSubscription?.unsubscribe();
  }

  /**
   * Load notifications from the service (manual refresh)
   */
  loadNotifications(): void {
    if (this.isRefreshing) return; // Prevent multiple simultaneous refreshes
    
    this.isRefreshing = true;
    // Use forceRefresh for manual refresh
    this.notificationService.forceRefresh();
    
    // Reset refreshing flag after a short delay
    setTimeout(() => {
      this.isRefreshing = false;
    }, 1000);
  }
  
  /**
   * Show confirmation modal for clearing all notifications
   */
  showClearConfirm(): void {
    if (this.notifications.length === 0) return;
    this.showClearConfirmModal = true;
  }

  /**
   * Close confirmation modal
   */
  closeClearConfirm(): void {
    this.showClearConfirmModal = false;
  }

  /**
   * Clear all notifications (called after confirmation)
   */
  clearAllNotifications(): void {
    if (this.isClearing || this.notifications.length === 0) {
      console.log('Clear all blocked - isClearing:', this.isClearing, 'count:', this.notifications.length);
      return;
    }
    
    console.log('Starting clear all - notifications count:', this.notifications.length);
    
    this.isClearing = true;
    this.showClearConfirmModal = false; // Close modal immediately
    
    // Clear local state immediately for instant UI update
    this.notifications = [];
    this.unreadCount = 0;
    this.newNotifications = [];
    this.showNewNotifications = false;
    
    console.log('Local state cleared, calling deleteAllNotifications API...');
    
    this.notificationService.deleteAllNotifications().subscribe({
      next: (response) => {
        console.log('Delete all response in component:', response);
        
        // Verify deletion was successful
        if (response && (response.success !== false && (response.deleted_count >= 0 || response.success === true))) {
          // State is already cleared above, just confirm it worked
          console.log('Deletion successful, keeping UI cleared');
          
          // Wait a bit before allowing polling to resume
          setTimeout(() => {
            this.isClearing = false;
            console.log('Clear operation complete, polling will resume');
          }, 3000); // Wait 3 seconds to ensure backend deletion is complete
          
          // Close panel if open
          this.closePanel();
        } else {
          // If deletion didn't work, refresh to restore state
          console.warn('Deletion may have failed, refreshing...', response);
          this.isClearing = false;
          setTimeout(() => {
            this.notificationService.forceRefresh();
          }, 500);
        }
      },
      error: (error) => {
        console.error('Error clearing notifications in component:', error);
        console.error('Error details:', {
          status: error.status,
          statusText: error.statusText,
          error: error.error,
          url: error.url
        });
        
        this.isClearing = false;
        
        // On error, refresh to get actual state from backend
        // This will restore notifications if deletion failed
        console.log('Refreshing notifications after error...');
        setTimeout(() => {
          this.notificationService.forceRefresh();
        }, 500);
      }
    });
  }

  /**
   * Toggle notification panel
   */
  togglePanel(): void {
    this.isPanelOpen = !this.isPanelOpen;
    if (this.isPanelOpen) {
      // Mark all as read when opening panel
      this.markAllAsRead();
    }
  }

  /**
   * Close notification panel
   */
  closePanel(): void {
    this.isPanelOpen = false;
  }

  /**
   * Handle notification click
   */
  onNotificationClick(notification: Notification): void {
    // Mark as read
    if (!notification.is_read) {
      this.notificationService.markAsRead(notification.id).subscribe();
    }

    // Navigate if action URL is provided
    if (notification.action_url) {
      this.router.navigateByUrl(notification.action_url);
      this.closePanel();
    }
  }

  /**
   * Mark a notification as read
   */
  markAsRead(notification: Notification, event: Event): void {
    event.stopPropagation();
    if (!notification.is_read) {
      this.notificationService.markAsRead(notification.id).subscribe();
    }
  }

  /**
   * Mark all notifications as read
   */
  markAllAsRead(): void {
    if (this.unreadCount > 0) {
      this.notificationService.markAllAsRead().subscribe();
    }
  }

  /**
   * Delete a notification
   */
  deleteNotification(notification: Notification, event: Event): void {
    event.stopPropagation();
    this.notificationService.deleteNotification(notification.id).subscribe();
  }

  /**
   * Dismiss new notification slide-in
   */
  dismissNewNotification(notification: Notification): void {
    this.newNotifications = this.newNotifications.filter(n => n.id !== notification.id);
    if (this.newNotifications.length === 0) {
      this.showNewNotifications = false;
    }
  }

  /**
   * Get notification icon based on type
   */
  getNotificationIcon(type: string): string {
    switch (type) {
      case 'success':
        return 'fa-check-circle';
      case 'warning':
        return 'fa-exclamation-triangle';
      case 'error':
        return 'fa-times-circle';
      case 'assignment':
        return 'fa-clipboard-list';
      case 'approval':
        return 'fa-thumbs-up';
      default:
        return 'fa-info-circle';
    }
  }

  /**
   * Get notification color class based on type
   */
  getNotificationColorClass(type: string): string {
    switch (type) {
      case 'success':
        return 'bg-green-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'error':
        return 'bg-red-500';
      case 'assignment':
        return 'bg-blue-500';
      case 'approval':
        return 'bg-emerald-500';
      default:
        return 'bg-sky-500';
    }
  }

  /**
   * Format relative time (e.g., "2 minutes ago")
   */
  getRelativeTime(dateString: string): string {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) {
      return 'Just now';
    } else if (diffMins < 60) {
      return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    } else if (diffHours < 24) {
      return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    } else if (diffDays < 7) {
      return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    } else {
      return date.toLocaleDateString();
    }
  }

  /**
   * Close panel when clicking outside
   */
  @HostListener('document:click', ['$event'])
  onClickOutside(event: Event): void {
    if (this.isPanelOpen) {
      const target = event.target as HTMLElement;
      if (this.notificationPanel && !this.notificationPanel.nativeElement.contains(target)) {
        if (this.bellIcon && !this.bellIcon.nativeElement.contains(target)) {
          this.closePanel();
        }
      }
    }
  }
}

