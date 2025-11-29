import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';

/**
 * Centralized API service for managing all API endpoints
 * This ensures consistency across the application and makes it easy to
 * switch between development and production environments
 */
@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly baseUrl: string;

  constructor() {
    // Use environment configuration, fallback to localhost if not set
    this.baseUrl = environment.apiUrl || 'http://localhost:8000';
  }

  /**
   * Get the full URL for an API endpoint
   * @param endpoint - The endpoint path (e.g., '/trainings/', '/login')
   * @returns Full URL string
   */
  getUrl(endpoint: string): string {
    // Ensure endpoint starts with /
    const normalizedEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    return `${this.baseUrl}${normalizedEndpoint}`;
  }

  // Authentication endpoints
  get loginUrl(): string {
    return this.getUrl('/login');
  }

  get registerUrl(): string {
    return this.getUrl('/register');
  }

  // Dashboard endpoints
  get engineerDashboardUrl(): string {
    return this.getUrl('/data/engineer');
  }

  get managerDashboardUrl(): string {
    return this.getUrl('/data/manager/dashboard');
  }

  // Training endpoints
  get trainingsUrl(): string {
    return this.getUrl('/trainings/');
  }

  get myTrainingsUrl(): string {
    return this.getUrl('/trainings/my-trainings');
  }

  trainingUrl(id: number): string {
    return this.getUrl(`/trainings/${id}`);
  }

  // Assignment endpoints
  get assignmentsUrl(): string {
    return this.getUrl('/assignments/');
  }

  get myAssignmentsUrl(): string {
    return this.getUrl('/assignments/my');
  }

  get managerTeamAssignmentsUrl(): string {
    return this.getUrl('/assignments/manager/team');
  }

  getTrainingCandidatesUrl(trainingId: number): string {
    return this.getUrl(`/assignments/training/${trainingId}/candidates`);
  }

  markTrainingAttendanceUrl(trainingId: number): string {
    return this.getUrl(`/assignments/training/${trainingId}/attendance`);
  }

  // Training request endpoints
  get trainingRequestsUrl(): string {
    return this.getUrl('/training-requests/');
  }

  get myTrainingRequestsUrl(): string {
    return this.getUrl('/training-requests/my');
  }

  get pendingTrainingRequestsUrl(): string {
    return this.getUrl('/training-requests/pending');
  }

  trainingRequestRespondUrl(id: number): string {
    return this.getUrl(`/training-requests/${id}/respond`);
  }

  // Additional skills endpoints
  get additionalSkillsUrl(): string {
    return this.getUrl('/additional-skills/');
  }

  additionalSkillUrl(id: number): string {
    return this.getUrl(`/additional-skills/${id}`);
  }

  // Shared content endpoints
  get sharedAssignmentsUrl(): string {
    return this.getUrl('/shared-content/assignments');
  }

  sharedAssignmentUrl(trainingId: number): string {
    return this.getUrl(`/shared-content/assignments/${trainingId}`);
  }

  sharedAssignmentResultUrl(trainingId: number): string {
    return this.getUrl(`/shared-content/assignments/${trainingId}/result`);
  }

  get sharedAssignmentSubmitUrl(): string {
    return this.getUrl('/shared-content/assignments/submit');
  }

  trainerAssignmentsUrl(trainingId: number): string {
    return this.getUrl(`/shared-content/trainer/assignments/${trainingId}`);
  }

  get sharedFeedbackUrl(): string {
    return this.getUrl('/shared-content/feedback');
  }

  sharedFeedbackUrlById(trainingId: number): string {
    return this.getUrl(`/shared-content/feedback/${trainingId}`);
  }

  get sharedFeedbackSubmitUrl(): string {
    return this.getUrl('/shared-content/feedback/submit');
  }

  trainerFeedbackUrl(trainingId: number): string {
    return this.getUrl(`/shared-content/trainer/feedback/${trainingId}`);
  }

  get managerTeamAssignmentsSubmissionsUrl(): string {
    return this.getUrl('/shared-content/manager/team/assignments');
  }

  get managerTeamFeedbackSubmissionsUrl(): string {
    return this.getUrl('/shared-content/manager/team/feedback');
  }

  get managerPerformanceFeedbackUrl(): string {
    return this.getUrl('/shared-content/manager/performance-feedback');
  }

  managerPerformanceFeedbackByIdUrl(trainingId: number, employeeId: string): string {
    return this.getUrl(`/shared-content/manager/performance-feedback/${trainingId}/${employeeId}`);
  }

  managerPerformanceFeedbackHistoryUrl(trainingId: number, employeeId: string): string {
    return this.getUrl(`/shared-content/manager/performance-feedback/${trainingId}/${employeeId}/history`);
  }

  get employeePerformanceFeedbackUrl(): string {
    return this.getUrl('/shared-content/employee/performance-feedback');
  }

  // Manager team skill update
  get managerTeamSkillUpdateUrl(): string {
    return this.getUrl('/data/manager/team-skill');
  }

  // Training files endpoints
  get uploadQuestionFileUrl(): string {
    return this.getUrl('/training-files/questions/upload');
  }

  questionFileUrl(trainingId: number): string {
    return this.getUrl(`/training-files/questions/${trainingId}`);
  }

  get uploadSolutionFileUrl(): string {
    return this.getUrl('/training-files/solutions/upload');
  }

  solutionFileUrl(trainingId: number, employeeId: string): string {
    return this.getUrl(`/training-files/solutions/${trainingId}/${employeeId}`);
  }

  trainerSolutionsUrl(trainingId: number): string {
    return this.getUrl(`/training-files/trainer/solutions/${trainingId}`);
  }

  // Notification endpoints
  get notificationsUrl(): string {
    return this.getUrl('/notifications/');
  }

  get unreadCountUrl(): string {
    return this.getUrl('/notifications/unread-count');
  }

  markNotificationReadUrl(id: number): string {
    return this.getUrl(`/notifications/${id}/read`);
  }

  get markAllNotificationsReadUrl(): string {
    return this.getUrl('/notifications/read-all');
  }

  deleteNotificationUrl(id: number): string {
    return this.getUrl(`/notifications/${id}`);
  }

  get deleteAllNotificationsUrl(): string {
    return this.getUrl('/notifications/all');
  }
}
