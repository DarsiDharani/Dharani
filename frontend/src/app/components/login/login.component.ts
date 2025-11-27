/**
 * Login Component
 * 
 * Purpose: Handles user authentication and login functionality
 * Features:
 * - Form-based login with username and password
 * - Validates username as numeric employee ID
 * - Displays success/error popups for user feedback
 * - Redirects to appropriate dashboard based on user role (manager/employee)
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { AuthService } from '../../services/auth.service';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
})
export class LoginComponent implements OnInit {
  /** Reactive form for login credentials */
  loginForm!: FormGroup;
  
  /** Flag to track if form has been submitted (for validation display) */
  isSubmitted = false;
  
  /** Flag to show loading state during API call */
  isLoading = false;

  // Pop-up state management
  /** Controls visibility of popup modal */
  showPopup = false;
  
  /** Type of popup: 'success' for successful login, 'error' for failures */
  popupType: 'success' | 'error' = 'success';
  
  /** Message content to display in popup */
  popupMessage = '';
  
  /** Title text for popup header */
  popupTitle = '';

  /**
   * Component constructor - injects required services
   * @param fb - FormBuilder for creating reactive forms
   * @param router - Router for navigation
   * @param http - HttpClient for API calls
   * @param authService - Service for authentication operations
   * @param apiService - Service for API endpoint management
   */
  constructor(
    private fb: FormBuilder,
    private router: Router,
    private http: HttpClient,
    private authService: AuthService,
    private apiService: ApiService
  ) {}

  /**
   * Angular lifecycle hook - initializes the component
   * Sets up the login form with validators:
   * - Username: Required, must be numeric (employee ID)
   * - Password: Required
   */
  ngOnInit(): void {
    this.loginForm = this.fb.group({
      username: ['', [Validators.required, Validators.pattern('^[0-9]+$')]],
      password: ['', [Validators.required]],
    });
  }

  /**
   * Getter for easy access to form controls in template
   * @returns Form controls object for validation display
   */
  get f() {
    return this.loginForm.controls;
  }

  /**
   * Handles form submission
   * - Validates form
   * - Sends login request to API
   * - Stores authentication token and user info on success
   * - Redirects to appropriate dashboard based on role
   * - Shows error popup on failure
   */
  onSubmit(): void {
    this.isSubmitted = true;
    if (this.loginForm.invalid) return;

    this.isLoading = true;

    const body = {
      username: this.loginForm.value.username,
      password: this.loginForm.value.password,
    };

    this.http
      .post<any>(this.apiService.loginUrl, body, {
        headers: new HttpHeaders({
          'Content-Type': 'application/json',
        }),
      })
      .subscribe({
        next: (response) => {
          this.authService.login(
            response.access_token,
            response.role,
            this.loginForm.value.username
          );

          this.showSuccessPopup('Login Successful!', 'Redirecting to your dashboard...');
          
          setTimeout(() => {
            if (response.role === 'manager') {
              this.router.navigate(['/manager-dashboard']);
            } else if (response.role === 'employee') {
              this.router.navigate(['/engineer-dashboard']);
            } else {
              this.showErrorPopup('Login Failed', 'Unknown role received.');
            }
          }, 1500); // Wait for the pop-up to be seen
          
          this.isLoading = false;
        },
        error: (err) => {
          this.showErrorPopup('Login Failed', err.error?.detail || 'Login failed. Please try again.');
          this.isLoading = false;
        },
      });
  }

  /**
   * Displays success popup with custom title and message
   * @param title - Popup header text
   * @param message - Popup body message
   */
  showSuccessPopup(title: string, message: string): void {
    this.popupType = 'success';
    this.popupTitle = title;
    this.popupMessage = message;
    this.showPopup = true;
  }

  /**
   * Displays error popup with custom title and message
   * @param title - Popup header text
   * @param message - Popup body message (usually error details from API)
   */
  showErrorPopup(title: string, message: string): void {
    this.popupType = 'error';
    this.popupTitle = title;
    this.popupMessage = message;
    this.showPopup = true;
  }

  /**
   * Closes the popup modal
   * Called when user clicks close button or outside the modal
   */
  closePopup(): void {
    this.showPopup = false;
  }
}