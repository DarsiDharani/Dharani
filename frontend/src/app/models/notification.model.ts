/**
 * Notification Model Interfaces
 * 
 * Purpose: TypeScript interfaces for notification-related data structures
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

/**
 * Notification interface representing a user notification
 */
export interface Notification {
    /** Unique notification ID */
    id: number;
    /** Employee ID of the user receiving the notification */
    user_empid: string;
    /** Notification title/heading */
    title: string;
    /** Notification message content */
    message: string;
    /** Notification type (info, success, warning, error, assignment, approval, etc.) */
    type: string;
    /** Whether the notification has been read */
    is_read: boolean;
    /** Optional ID of related entity (training_id, request_id, etc.) */
    related_id?: number;
    /** Optional type of related entity (training, request, assignment, etc.) */
    related_type?: string;
    /** Optional URL to navigate to when notification is clicked */
    action_url?: string;
    /** Notification creation timestamp */
    created_at: string;
    /** Timestamp when notification was marked as read */
    read_at?: string;
}

/**
 * Unread count response interface
 */
export interface UnreadCountResponse {
    unread_count: number;
}



