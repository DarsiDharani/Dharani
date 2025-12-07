/**
 * Admin Dashboard Component - COMPLETE IMPLEMENTATION
 * 
 * Main admin tasks:
 * 1. User Management - CRUD operations for all users
 * 2. Training Management - Create, edit, delete trainings system-wide
 * 3. Skills Management - View and edit all employee skills
 * 4. Data Management - Import Excel/CSV files
 * 5. Analytics - System-wide statistics
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { ApiService } from '../../services/api.service';
import { trigger, style, animate, transition } from '@angular/animations';
import { ToastService } from '../../services/toast.service';

interface User {
  username: string;
  name: string;
  role: string;
  is_trainer: boolean;
  manager_name?: string;
  created_at?: string;
}

interface Training {
  id: number;
  training_name: string;
  trainer_name: string;
  training_date?: string;
  skill?: string;
  assigned_count: number;
  attended_count: number;
  completion_rate: number;
}

interface Competency {
  id: number;
  employee_empid: string;
  employee_name: string;
  skill: string;
  competency: string;
  current_expertise: string;
  target_expertise: string;
  status: string;
  department?: string;
}

interface GapAnalysis {
  total_skills: number;
  skills_met: number;
  skills_gap: number;
  gap_percentage: number;
}

@Component({
  selector: 'app-admin-dashboard',
  templateUrl: './admin-dashboard.component.html',
  styleUrls: ['./admin-dashboard.component.css'],
  animations: [
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('500ms ease-in', style({ opacity: 1 }))
      ]),
      transition(':leave', [
        animate('500ms ease-out', style({ opacity: 0 }))
      ])
    ]),
    trigger('slideFadeIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(-20px)' }),
        animate('500ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ])
    ]),
    trigger('bouncyScale', [
      transition(':enter', [
        style({ opacity: 0, transform: 'scale(0.8)' }),
        animate('600ms cubic-bezier(0.68, -0.55, 0.265, 1.55)', style({ opacity: 1, transform: 'scale(1)' }))
      ])
    ])
  ]
})
export class AdminDashboardComponent implements OnInit {
  // Component state
  activeTab: string = 'dashboard';
  adminName: string = 'Admin';
  adminId: string = '';
  
  // Dashboard metrics
  totalUsers: number = 0;
  totalManagers: number = 0;
  totalEmployees: number = 0;
  totalTrainings: number = 0;
  totalAssignments: number = 0;
  totalSkills: number = 0;
  pendingRequests: number = 0;
  activeTrainers: number = 0;
  
  // User Management
  users: User[] = [];
  usersLoading: boolean = false;
  userSearch: string = '';
  userRoleFilter: string = 'all';
  showCreateUserModal: boolean = false;
  showEditUserModal: boolean = false;
  selectedUser: User | null = null;
  selectedUsers: Set<string> = new Set<string>(); // Track selected usernames for bulk delete
  isSelectAll: boolean = false;
  newUser: any = {
    username: '',
    password: '',
    name: '',
    role: 'employee',
    manager_empid: '',
    is_trainer: false
  };
  
  // Training Management
  trainings: Training[] = [];
  trainingsLoading: boolean = false;
  trainingSearch: string = '';
  trainingSkillFilter: string = '';
  showCreateTrainingModal: boolean = false;
  showEditTrainingModal: boolean = false;
  selectedTraining: Training | null = null;
  selectedTrainings: Set<number> = new Set<number>(); // Track selected training IDs for bulk delete
  isSelectAllTrainings: boolean = false;
  newTraining: any = {
    training_name: '',
    trainer_name: '',
    skill: '',
    competency: '',
    training_date: '',
    duration: '',
    time: '',
    training_type: '',
    seats: '',
    training_topics: '',
    prerequisites: '',
    assessment_details: ''
  };
  
  // Skills Management
  competencies: Competency[] = [];
  competenciesLoading: boolean = false;
  skillSearch: string = '';
  skillEmployeeFilter: string = '';
  showEditSkillModal: boolean = false;
  selectedCompetency: Competency | null = null;
  skillUpdate: any = {
    current_expertise: '',
    target_expertise: ''
  };
  gapAnalysis: GapAnalysis | null = null;
  
  // Data Management
  excelFile: File | null = null;
  csvFile: File | null = null;
  uploading: boolean = false;
  
  // Analytics
  analytics: any = null;
  analyticsLoading: boolean = false;
  
  isLoading: boolean = false;

  constructor(
    private http: HttpClient,
    private router: Router,
    private authService: AuthService,
    private apiService: ApiService,
    private toastService: ToastService
  ) {}

  ngOnInit(): void {
    const role = this.authService.getRole();
    if (role !== 'admin') {
      this.toastService.show('Access denied. Admin privileges required.', 'error');
      if (role === 'manager') {
        this.router.navigate(['/manager-dashboard']);
      } else {
        this.router.navigate(['/engineer-dashboard']);
      }
      return;
    }

    this.adminId = this.authService.getUsername() || '';
    this.adminName = `Admin (${this.adminId})`;
    
    this.loadDashboardData();
  }

  getHeaders(): HttpHeaders {
    const token = this.authService.getToken();
    if (!token) {
      console.warn('No authentication token available');
      // Return headers without Authorization if token is missing
      return new HttpHeaders({
        'Content-Type': 'application/json'
      });
    }
    return new HttpHeaders({
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    });
  }

  getFormHeaders(): HttpHeaders {
    const token = this.authService.getToken();
    return new HttpHeaders({
      'Authorization': `Bearer ${token}`
    });
  }

  // ==================== DASHBOARD ====================
  loadDashboardData(): void {
    // Check if user is logged in before making request
    if (!this.authService.isLoggedIn()) {
      this.toastService.show('Please log in to continue', 'error');
      this.router.navigate(['/login']);
      return;
    }

    this.isLoading = true;
    this.http.get<any>(this.apiService.adminDashboardUrl, { headers: this.getHeaders() })
      .subscribe({
        next: (data) => {
          this.adminName = data.admin_name || this.adminName;
          this.adminId = data.admin_id || this.adminId;
          
          if (data.metrics) {
            this.totalUsers = data.metrics.total_users || 0;
            this.totalManagers = data.metrics.total_managers || 0;
            this.totalEmployees = data.metrics.total_employees || 0;
            this.totalTrainings = data.metrics.total_trainings || 0;
            this.totalAssignments = data.metrics.total_assignments || 0;
            this.totalSkills = data.metrics.total_skills || 0;
            this.pendingRequests = data.metrics.pending_requests || 0;
            this.activeTrainers = data.metrics.active_trainers || 0;
          }
          
          this.isLoading = false;
        },
        error: (err) => {
          console.error('Error loading dashboard data:', err);
          // Handle 401 Unauthorized - redirect to login
          if (err.status === 401 || err.status === 403) {
            this.toastService.show('Session expired. Please log in again', 'error');
            this.authService.logout();
            this.router.navigate(['/login']);
          } else {
            this.toastService.show('Failed to load dashboard data', 'error');
          }
          this.isLoading = false;
        }
      });
  }

  // ==================== USER MANAGEMENT ====================
  loadUsers(): void {
    this.usersLoading = true;
    let url = `${this.apiService.adminUsersUrl}?page=1&limit=1000`;
    if (this.userRoleFilter !== 'all') {
      url += `&role=${this.userRoleFilter}`;
    }
    if (this.userSearch) {
      url += `&search=${this.userSearch}`;
    }
    
    this.http.get<any>(url, { headers: this.getHeaders() })
      .subscribe({
        next: (data) => {
          this.users = data.users || [];
          // Clear selections when users are reloaded (e.g., after search/filter)
          this.selectedUsers.clear();
          this.isSelectAll = false;
          this.usersLoading = false;
        },
        error: (err) => {
          console.error('Error loading users:', err);
          this.toastService.show('Failed to load users', 'error');
          this.usersLoading = false;
        }
      });
  }

  openCreateUserModal(): void {
    this.newUser = {
      username: '',
      password: '',
      name: '',
      role: 'employee',
      manager_empid: '',
      is_trainer: false
    };
    this.showCreateUserModal = true;
  }

  createUser(): void {
    if (!this.newUser.username || !this.newUser.password || !this.newUser.name) {
      this.toastService.show('Please fill all required fields', 'error');
      return;
    }

    this.http.post(this.apiService.adminUsersUrl, this.newUser, { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('User created successfully', 'success');
          this.showCreateUserModal = false;
          this.loadUsers();
          this.loadDashboardData();
          // Delay analytics refresh slightly to ensure database is updated
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 500);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to create user', 'error');
        }
      });
  }

  openEditUserModal(user: User): void {
    this.selectedUser = { ...user };
    this.showEditUserModal = true;
  }

  updateUser(): void {
    if (!this.selectedUser) return;

    const updateData: any = {
      name: this.selectedUser.name,
      role: this.selectedUser.role,
      is_trainer: this.selectedUser.is_trainer
    };

    this.http.put(this.apiService.adminUserUrl(this.selectedUser.username), updateData, { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('User updated successfully', 'success');
          this.showEditUserModal = false;
          this.loadUsers();
          this.loadDashboardData();
          // Delay analytics refresh slightly to ensure database is updated
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 500);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to update user', 'error');
        }
      });
  }

  deleteUser(user: User): void {
    if (!confirm(`Are you sure you want to delete user ${user.username}?`)) return;

    this.http.delete(this.apiService.adminUserUrl(user.username), { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('User deleted successfully', 'success');
          this.selectedUsers.delete(user.username); // Remove from selection if deleted
          this.loadUsers();
          this.loadDashboardData();
          // Delay analytics refresh slightly to ensure database is updated
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 500);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to delete user', 'error');
        }
      });
  }

  // ==================== BULK USER OPERATIONS ====================
  toggleUserSelection(user: User): void {
    if (this.selectedUsers.has(user.username)) {
      this.selectedUsers.delete(user.username);
    } else {
      this.selectedUsers.add(user.username);
    }
    this.updateSelectAllState();
  }

  toggleSelectAll(): void {
    if (this.isSelectAll) {
      this.selectedUsers.clear();
    } else {
      this.users.forEach(user => {
        this.selectedUsers.add(user.username);
      });
    }
    this.isSelectAll = !this.isSelectAll;
  }

  updateSelectAllState(): void {
    this.isSelectAll = this.users.length > 0 && this.selectedUsers.size === this.users.length;
  }

  isUserSelected(username: string): boolean {
    return this.selectedUsers.has(username);
  }

  getSelectedUsersCount(): number {
    return this.selectedUsers.size;
  }

  deleteSelectedUsers(): void {
    const count = this.selectedUsers.size;
    if (count === 0) {
      this.toastService.show('Please select at least one user to delete', 'error');
      return;
    }

    const usernames = Array.from(this.selectedUsers);
    if (!confirm(`Are you sure you want to delete ${count} user(s)?\n\nUsers: ${usernames.join(', ')}`)) {
      return;
    }

    // Delete users sequentially to avoid overwhelming the server
    let deletedCount = 0;
    let errorCount = 0;
    const total = usernames.length;

    usernames.forEach((username, index) => {
      this.http.delete(this.apiService.adminUserUrl(username), { headers: this.getHeaders() })
        .subscribe({
          next: () => {
            deletedCount++;
            this.selectedUsers.delete(username);
            
            // If this is the last request, refresh data
            if (deletedCount + errorCount === total) {
              if (errorCount > 0) {
                this.toastService.show(`Deleted ${deletedCount} user(s). ${errorCount} failed.`, 'warning');
              } else {
                this.toastService.show(`Successfully deleted ${deletedCount} user(s)`, 'success');
              }
              this.selectedUsers.clear();
              this.isSelectAll = false;
              this.loadUsers();
              this.loadDashboardData();
              // Delay analytics refresh slightly to ensure database is updated
              setTimeout(() => {
                this.loadAnalytics(); // Refresh analytics data
              }, 500);
            }
          },
          error: (err) => {
            errorCount++;
            console.error(`Failed to delete user ${username}:`, err);
            
            // If this is the last request, refresh data
            if (deletedCount + errorCount === total) {
              if (deletedCount > 0) {
                this.toastService.show(`Deleted ${deletedCount} user(s). ${errorCount} failed.`, 'warning');
              } else {
                this.toastService.show('Failed to delete users', 'error');
              }
              this.selectedUsers.clear();
              this.isSelectAll = false;
              this.loadUsers();
              this.loadDashboardData();
              // Delay analytics refresh slightly to ensure database is updated
              setTimeout(() => {
                this.loadAnalytics(); // Refresh analytics data
              }, 500);
            }
          }
        });
    });
  }

  resetPassword(user: User): void {
    const newPassword = prompt(`Enter new password for ${user.username}:`);
    if (!newPassword) return;

    this.http.post(this.apiService.adminResetPasswordUrl(user.username), { new_password: newPassword }, { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('Password reset successfully', 'success');
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to reset password', 'error');
        }
      });
  }

  // ==================== TRAINING MANAGEMENT ====================
  loadTrainings(): void {
    this.trainingsLoading = true;
    let url = this.apiService.adminTrainingsUrl;
    if (this.trainingSkillFilter) {
      url += `?skill=${this.trainingSkillFilter}`;
    }
    if (this.trainingSearch) {
      url += url.includes('?') ? `&trainer=${this.trainingSearch}` : `?trainer=${this.trainingSearch}`;
    }
    
    this.http.get<any>(url, { headers: this.getHeaders() })
      .subscribe({
        next: (data) => {
          this.trainings = data.trainings || [];
          // Clear selections when trainings are reloaded (e.g., after search/filter)
          this.selectedTrainings.clear();
          this.isSelectAllTrainings = false;
          this.trainingsLoading = false;
        },
        error: (err) => {
          console.error('Error loading trainings:', err);
          this.toastService.show('Failed to load trainings', 'error');
          this.trainingsLoading = false;
        }
      });
  }

  openCreateTrainingModal(): void {
    this.newTraining = {
      training_name: '',
      trainer_name: '',
      skill: '',
      competency: '',
      training_date: '',
      duration: '',
      time: '',
      training_type: '',
      seats: '',
      training_topics: '',
      prerequisites: '',
      assessment_details: ''
    };
    this.showCreateTrainingModal = true;
  }

  createTraining(): void {
    if (!this.newTraining.training_name || !this.newTraining.trainer_name) {
      this.toastService.show('Training name and trainer name are required', 'error');
      return;
    }

    this.http.post(this.apiService.adminTrainingsUrl, this.newTraining, { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('Training created successfully', 'success');
          this.showCreateTrainingModal = false;
          this.loadTrainings();
          this.loadDashboardData();
          // Delay analytics refresh slightly to ensure database is updated
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 500);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to create training', 'error');
        }
      });
  }

  openEditTrainingModal(training: Training): void {
    this.selectedTraining = { ...training };
    this.showEditTrainingModal = true;
  }

  updateTraining(): void {
    if (!this.selectedTraining) return;

    // Prepare update data from selectedTraining
    const updateData: any = {
      training_name: this.selectedTraining.training_name,
      trainer_name: this.selectedTraining.trainer_name,
      skill: this.selectedTraining.skill || '',
      training_date: this.selectedTraining.training_date || ''
    };

    this.http.put(this.apiService.adminTrainingUrl(this.selectedTraining.id), updateData, { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('Training updated successfully', 'success');
          this.showEditTrainingModal = false;
          this.loadTrainings();
          this.loadDashboardData();
          // Delay analytics refresh slightly to ensure database is updated
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 500);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to update training', 'error');
        }
      });
  }

  deleteTraining(training: Training): void {
    if (!confirm(`Are you sure you want to delete training "${training.training_name}"?`)) return;

    this.http.delete(this.apiService.adminTrainingUrl(training.id), { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('Training deleted successfully', 'success');
          this.selectedTrainings.delete(training.id); // Remove from selection if deleted
          this.loadTrainings();
          this.loadDashboardData();
          // Delay analytics refresh slightly to ensure database is updated
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 500);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to delete training', 'error');
        }
      });
  }

  // ==================== BULK TRAINING OPERATIONS ====================
  toggleTrainingSelection(training: Training): void {
    if (this.selectedTrainings.has(training.id)) {
      this.selectedTrainings.delete(training.id);
    } else {
      this.selectedTrainings.add(training.id);
    }
    this.updateSelectAllTrainingsState();
  }

  toggleSelectAllTrainings(): void {
    if (this.isSelectAllTrainings) {
      this.selectedTrainings.clear();
    } else {
      this.trainings.forEach(training => {
        this.selectedTrainings.add(training.id);
      });
    }
    this.isSelectAllTrainings = !this.isSelectAllTrainings;
  }

  updateSelectAllTrainingsState(): void {
    this.isSelectAllTrainings = this.trainings.length > 0 && this.selectedTrainings.size === this.trainings.length;
  }

  isTrainingSelected(trainingId: number): boolean {
    return this.selectedTrainings.has(trainingId);
  }

  getSelectedTrainingsCount(): number {
    return this.selectedTrainings.size;
  }

  deleteSelectedTrainings(): void {
    const count = this.selectedTrainings.size;
    if (count === 0) {
      this.toastService.show('Please select at least one training to delete', 'error');
      return;
    }

    const trainingIds = Array.from(this.selectedTrainings);
    const trainingNames = this.trainings
      .filter(t => trainingIds.includes(t.id))
      .map(t => t.training_name);
    
    if (!confirm(`Are you sure you want to delete ${count} training(s)?\n\nTrainings: ${trainingNames.join(', ')}`)) {
      return;
    }

    // Delete trainings sequentially to avoid overwhelming the server
    let deletedCount = 0;
    let errorCount = 0;
    const total = trainingIds.length;

    trainingIds.forEach((trainingId, index) => {
      this.http.delete(this.apiService.adminTrainingUrl(trainingId), { headers: this.getHeaders() })
        .subscribe({
          next: () => {
            deletedCount++;
            this.selectedTrainings.delete(trainingId);
            
            // If this is the last request, refresh data
            if (deletedCount + errorCount === total) {
              if (errorCount > 0) {
                this.toastService.show(`Deleted ${deletedCount} training(s). ${errorCount} failed.`, 'warning');
              } else {
                this.toastService.show(`Successfully deleted ${deletedCount} training(s)`, 'success');
              }
              this.selectedTrainings.clear();
              this.isSelectAllTrainings = false;
              this.loadTrainings();
              this.loadDashboardData();
              // Delay analytics refresh slightly to ensure database is updated
              setTimeout(() => {
                this.loadAnalytics(); // Refresh analytics data
              }, 500);
            }
          },
          error: (err) => {
            errorCount++;
            console.error(`Failed to delete training ${trainingId}:`, err);
            
            // If this is the last request, refresh data
            if (deletedCount + errorCount === total) {
              if (deletedCount > 0) {
                this.toastService.show(`Deleted ${deletedCount} training(s). ${errorCount} failed.`, 'warning');
              } else {
                this.toastService.show('Failed to delete trainings', 'error');
              }
              this.selectedTrainings.clear();
              this.isSelectAllTrainings = false;
              this.loadTrainings();
              this.loadDashboardData();
              // Delay analytics refresh slightly to ensure database is updated
              setTimeout(() => {
                this.loadAnalytics(); // Refresh analytics data
              }, 500);
            }
          }
        });
    });
  }

  // ==================== SKILLS MANAGEMENT ====================
  loadCompetencies(): void {
    this.competenciesLoading = true;
    let url = this.apiService.adminSkillsCompetenciesUrl;
    if (this.skillEmployeeFilter) {
      url += `?employee_empid=${this.skillEmployeeFilter}`;
    }
    if (this.skillSearch) {
      url += url.includes('?') ? `&skill=${this.skillSearch}` : `?skill=${this.skillSearch}`;
    }
    
    this.http.get<any>(url, { headers: this.getHeaders() })
      .subscribe({
        next: (data) => {
          this.competencies = data.competencies || [];
          this.competenciesLoading = false;
        },
        error: (err) => {
          console.error('Error loading competencies:', err);
          this.toastService.show('Failed to load competencies', 'error');
          this.competenciesLoading = false;
        }
      });
  }

  loadGapAnalysis(): void {
    this.http.get<any>(this.apiService.adminSkillsGapAnalysisUrl, { headers: this.getHeaders() })
      .subscribe({
        next: (data) => {
          this.gapAnalysis = data;
        },
        error: (err) => {
          console.error('Error loading gap analysis:', err);
          this.toastService.show('Failed to load gap analysis', 'error');
        }
      });
  }

  openEditSkillModal(competency: Competency): void {
    this.selectedCompetency = { ...competency };
    this.skillUpdate = {
      current_expertise: competency.current_expertise,
      target_expertise: competency.target_expertise
    };
    this.showEditSkillModal = true;
  }

  updateSkill(): void {
    if (!this.selectedCompetency) return;

    this.http.put(this.apiService.adminSkillCompetencyUrl(this.selectedCompetency.id), this.skillUpdate, { headers: this.getHeaders() })
      .subscribe({
        next: () => {
          this.toastService.show('Skill updated successfully', 'success');
          this.showEditSkillModal = false;
          this.loadCompetencies();
          this.loadGapAnalysis();
          this.loadDashboardData();
          // Delay analytics refresh slightly to ensure database is updated
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 500);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to update skill', 'error');
        }
      });
  }

  // ==================== DATA MANAGEMENT ====================
  onExcelFileSelected(event: any): void {
    this.excelFile = event.target.files[0];
  }

  onCsvFileSelected(event: any): void {
    this.csvFile = event.target.files[0];
  }

  uploadExcel(): void {
    if (!this.excelFile) {
      this.toastService.show('Please select an Excel file', 'error');
      return;
    }

    this.uploading = true;
    const formData = new FormData();
    formData.append('file', this.excelFile);

    this.http.post(this.apiService.uploadExcelUrl, formData, { headers: this.getFormHeaders() })
      .subscribe({
        next: (data: any) => {
          this.toastService.show(data.message || 'Excel file uploaded successfully', 'success');
          this.uploading = false;
          this.excelFile = null;
          // Refresh all data to ensure consistency
          this.loadDashboardData();
          this.loadUsers();
          this.loadTrainings();
          this.loadCompetencies();
          this.loadGapAnalysis();
          // Delay analytics refresh to ensure database is fully updated after file upload
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 1000);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to upload Excel file', 'error');
          this.uploading = false;
        }
      });
  }

  uploadCsv(): void {
    if (!this.csvFile) {
      this.toastService.show('Please select a CSV file', 'error');
      return;
    }

    this.uploading = true;
    const formData = new FormData();
    formData.append('file', this.csvFile);

    this.http.post(this.apiService.uploadCsvUrl, formData, { headers: this.getFormHeaders() })
      .subscribe({
        next: (data: any) => {
          this.toastService.show(data.message || 'CSV file uploaded successfully', 'success');
          this.uploading = false;
          this.csvFile = null;
          // Refresh all data to ensure consistency
          this.loadDashboardData();
          this.loadUsers();
          // Delay analytics refresh to ensure database is fully updated after file upload
          setTimeout(() => {
            this.loadAnalytics(); // Refresh analytics data
          }, 1000);
        },
        error: (err) => {
          this.toastService.show(err.error?.detail || 'Failed to upload CSV file', 'error');
          this.uploading = false;
        }
      });
  }

  // ==================== ANALYTICS ====================
  loadAnalytics(): void {
    this.analyticsLoading = true;
    // Add cache-busting parameter to ensure fresh data
    const url = `${this.apiService.adminAnalyticsOverviewUrl}?t=${Date.now()}`;
    this.http.get<any>(url, { headers: this.getHeaders() })
      .subscribe({
        next: (data) => {
          this.analytics = data;
          this.analyticsLoading = false;
        },
        error: (err) => {
          console.error('Error loading analytics:', err);
          this.toastService.show('Failed to load analytics', 'error');
          this.analyticsLoading = false;
        }
      });
  }

  // ==================== TAB MANAGEMENT ====================
  selectTab(tab: string): void {
    this.activeTab = tab;
    
    // Load data when switching tabs
    if (tab === 'users') {
      this.loadUsers();
    } else if (tab === 'trainings') {
      this.loadTrainings();
    } else if (tab === 'skills') {
      this.loadCompetencies();
      this.loadGapAnalysis();
    } else if (tab === 'analytics') {
      // Always reload analytics when switching to analytics tab to ensure fresh data
      this.loadAnalytics();
    } else if (tab === 'dashboard') {
      // Reload dashboard data when switching back to dashboard
      this.loadDashboardData();
    }
  }

  logout(): void {
    this.authService.logout();
    this.router.navigate(['/login']);
  }
}
