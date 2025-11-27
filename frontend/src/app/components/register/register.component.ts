/**
 * Register Component
 * 
 * Purpose: Handles new user registration
 * Features:
 * - Employee ID validation (7-digit numeric)
 * - Password strength validation (min 6 chars, uppercase, lowercase, number)
 * - Password confirmation matching
 * - Form validation with user-friendly error messages
 * - Success/error popup notifications
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators, AbstractControl, ValidationErrors } from '@angular/forms';
import { Router } from '@angular/router';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { ApiService } from '../../services/api.service';

@Component({
  selector: 'app-register',
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.css'],
})
export class RegisterComponent implements OnInit {
  /** Reactive form for registration data */
  registerForm!: FormGroup;
  
  /** Flag to track if form has been submitted (for validation display) */
  isSubmitted = false;

  // Popup state management
  /** Controls visibility of popup modal */
  showPopup = false;
  
  /** Type of popup: 'success' for successful registration, 'error' for failures */
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
   * @param apiService - Service for API endpoint management
   */
  constructor(
    private fb: FormBuilder,
    private router: Router,
    private http: HttpClient,
    private apiService: ApiService
  ) {}

  /**
   * Angular lifecycle hook - initializes the component
   * Sets up the registration form with validators:
   * - Employee ID: Required, exactly 7 digits
   * - Password: Required, min 6 chars, must contain uppercase, lowercase, and number
   * - Confirm Password: Required, must match password field
   */
  ngOnInit(): void {
    this.registerForm = this.fb.group(
      {
        emp_id: ['', [Validators.required, Validators.pattern('^[0-9]{7}$')]],
        password: [
          '',
          [
            Validators.required,
            Validators.minLength(6),
            Validators.pattern('^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d).+$'),
          ],
        ],
        confirm_password: ['', Validators.required],
      },
      { validators: this.passwordMatchValidator }
    );
  }

  /**
   * Custom validator to ensure password and confirm_password fields match
   * @param form - The form group containing password fields
   * @returns ValidationErrors object if passwords don't match, null if they match
   */
  passwordMatchValidator(form: AbstractControl): ValidationErrors | null {
    const password = form.get('password')?.value;
    const confirmPassword = form.get('confirm_password')?.value;
    return password === confirmPassword ? null : { mismatch: true };
  }

  /**
   * Handles form submission
   * - Validates form (including password match)
   * - Sends registration request to API
   * - Shows success popup and redirects to login on success
   * - Shows error popup on failure
   * Note: confirm_password is not sent to backend (only used for validation)
   */
  onSubmit(): void {
    this.isSubmitted = true;
    if (this.registerForm.invalid) {
      console.warn('❌ Invalid form:', this.registerForm);
      // Debug logging for form validation errors (development only)
      console.log('Form Status:', this.registerForm.status);
      Object.keys(this.registerForm.controls).forEach(key => {
        const controlErrors = this.registerForm.get(key)?.errors;
        if (controlErrors != null) {
          console.log(`Control: ${key}, Errors:`, controlErrors);
        }
      });
      if (this.registerForm.errors) {
        console.log('Form-level errors:', this.registerForm.errors);
      }
      return;
    }

    // Payload for API: only send emp_id and password (confirm_password is for UI validation only)
    const registerData = {
      emp_id: this.registerForm.get('emp_id')?.value,
      password: this.registerForm.get('password')?.value,
    };

    this.http.post<any>(this.apiService.registerUrl, registerData).subscribe({
      next: () => {
        this.showSuccessPopup('Registration Successful!', 'Your account has been created successfully. Please login to continue.');
        setTimeout(() => {
          this.router.navigate(['/login']);
        }, 2000);
      },
      error: (err: HttpErrorResponse) => {
        console.error('❌ Registration API error:', err);
        this.showErrorPopup('Registration Failed', err.error.detail || 'Registration failed. Please try again.');
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