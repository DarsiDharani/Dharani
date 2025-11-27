/**
 * Toast Component
 * 
 * Purpose: Displays toast notifications to users
 * Features:
 * - Multiple toast messages support
 * - Auto-dismiss after duration
 * - Manual dismiss via close button
 * - Slide-in/out animations
 * - Different styles for success, error, warning, and info types
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit, OnDestroy } from '@angular/core';
import { ToastService, ToastMessage } from '../../services/toast.service';
import { trigger, transition, style, animate } from '@angular/animations';

@Component({
  selector: 'app-toast',
  templateUrl: './toast.component.html',
  styleUrls: ['./toast.component.css'],
  animations: [
    trigger('modalScale', [
      transition(':enter', [
        style({ transform: 'scale(0.8)', opacity: 0 }),
        animate('300ms cubic-bezier(0.34, 1.56, 0.64, 1)', style({ transform: 'scale(1)', opacity: 1 }))
      ]),
      transition(':leave', [
        animate('200ms ease-in', style({ transform: 'scale(0.9)', opacity: 0 }))
      ])
    ])
  ]
})
export class ToastComponent implements OnInit, OnDestroy {
  /** Array of active toast messages to display */
  toasts: ToastMessage[] = [];

  /**
   * Component constructor - injects ToastService
   * @param toastService - Service for managing toast notifications
   */
  constructor(private toastService: ToastService) {}

  /**
   * Angular lifecycle hook - subscribes to toast service to receive new toasts
   */
  ngOnInit(): void {
    this.toastService.toasts$.subscribe(toasts => {
      this.toasts = toasts;
    });
  }

  /**
   * Angular lifecycle hook - cleanup
   * Note: Subscription cleanup is handled automatically by Angular
   */
  ngOnDestroy(): void {
    // Cleanup handled by service
  }

  /**
   * Removes a toast message by ID
   * @param id - Unique identifier of the toast to remove
   */
  removeToast(id: string): void {
    this.toastService.remove(id);
  }

  /**
   * Returns the appropriate Font Awesome icon class based on toast type
   * @param type - Toast type: 'success', 'error', 'warning', or 'info'
   * @returns Font Awesome icon class name
   */
  getIconClass(type: string): string {
    switch (type) {
      case 'success': return 'fa-check-circle';
      case 'error': return 'fa-exclamation-circle';
      case 'warning': return 'fa-exclamation-triangle';
      case 'info': return 'fa-info-circle';
      default: return 'fa-bell';
    }
  }

  /**
   * Returns Tailwind CSS classes for toast container based on type
   * @param type - Toast type: 'success', 'error', 'warning', or 'info'
   * @returns Tailwind CSS class string for background, border, and text colors
   */
  getToastClass(type: string): string {
    switch (type) {
      case 'success': return 'bg-green-50 border-green-200 text-green-800';
      case 'error': return 'bg-red-50 border-red-200 text-red-800';
      case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'info': return 'bg-blue-50 border-blue-200 text-blue-800';
      default: return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  }

  /**
   * Returns Tailwind CSS classes for icon color based on toast type
   * @param type - Toast type: 'success', 'error', 'warning', or 'info'
   * @returns Tailwind CSS class string for icon text color
   */
  getIconColorClass(type: string): string {
    switch (type) {
      case 'success': return 'text-green-500';
      case 'error': return 'text-red-500';
      case 'warning': return 'text-yellow-500';
      case 'info': return 'text-blue-500';
      default: return 'text-gray-500';
    }
  }

  /**
   * Returns gradient classes for header based on toast type
   */
  getHeaderGradient(type: string): string {
    switch (type) {
      case 'success': return 'bg-gradient-to-r from-green-500 to-emerald-600';
      case 'error': return 'bg-gradient-to-r from-red-500 to-rose-600';
      case 'warning': return 'bg-gradient-to-r from-amber-500 to-orange-600';
      case 'info': return 'bg-gradient-to-r from-blue-500 to-cyan-600';
      default: return 'bg-gradient-to-r from-slate-500 to-slate-600';
    }
  }

  /**
   * Returns border classes based on toast type
   */
  getBorderClass(type: string): string {
    switch (type) {
      case 'success': return 'border-green-200';
      case 'error': return 'border-red-200';
      case 'warning': return 'border-amber-200';
      case 'info': return 'border-blue-200';
      default: return 'border-slate-200';
    }
  }

  /**
   * Returns button classes based on toast type
   */
  getButtonClass(type: string): string {
    switch (type) {
      case 'success': return 'bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-700 hover:to-emerald-700';
      case 'error': return 'bg-gradient-to-r from-red-600 to-rose-600 text-white hover:from-red-700 hover:to-rose-700';
      case 'warning': return 'bg-gradient-to-r from-amber-600 to-orange-600 text-white hover:from-amber-700 hover:to-orange-700';
      case 'info': return 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-700 hover:to-cyan-700';
      default: return 'bg-gradient-to-r from-slate-600 to-slate-700 text-white hover:from-slate-700 hover:to-slate-800';
    }
  }

  /**
   * Returns button icon based on toast type
   */
  getButtonIcon(type: string): string {
    switch (type) {
      case 'success': return 'fa-check';
      case 'error': return 'fa-times';
      case 'warning': return 'fa-exclamation';
      case 'info': return 'fa-info';
      default: return 'fa-check';
    }
  }

  /**
   * Returns button text based on toast type
   */
  getButtonText(type: string): string {
    switch (type) {
      case 'success': return 'Got it!';
      case 'error': return 'Okay';
      case 'warning': return 'Understood';
      case 'info': return 'Continue';
      default: return 'Close';
    }
  }

  /**
   * Formats message with line breaks and basic HTML
   */
  formatMessage(message: string): string {
    // Convert newlines to <br> and escape HTML
    return message
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>');
  }

  /**
   * Closes the current (first) toast
   */
  closeCurrentToast(): void {
    if (this.toasts.length > 0) {
      this.removeToast(this.toasts[0].id);
    }
  }
}
