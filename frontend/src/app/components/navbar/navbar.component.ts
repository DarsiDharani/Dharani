/**
 * Navbar Component
 * 
 * Purpose: Provides navigation bar with authentication-aware menu
 * Features:
 * - Responsive mobile menu toggle
 * - Authentication status checking
 * - Logout functionality
 * - Navigation to different routes
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-navbar',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})
export class NavbarComponent {
  /** Flag to control mobile menu visibility (hamburger menu) */
  isMenuOpen = false;

  /**
   * Component constructor - injects required services
   * @param router - Router for navigation (public for template access)
   * @param authService - Service for authentication operations
   */
  constructor(
    public router: Router,
    private authService: AuthService
  ) {}

  /**
   * Toggles the mobile menu open/closed state
   * Used for responsive navigation on smaller screens
   */
  toggleMenu() {
    this.isMenuOpen = !this.isMenuOpen;
  }

  /**
   * Checks if the user is logged in by calling the method from your AuthService.
   * @returns boolean
   */
  isLoggedIn(): boolean {
    return this.authService.isLoggedIn();
  }

  /**
   * Logs the user out and navigates to the login page.
   */
  logout(): void {
    this.authService.logout();
    this.router.navigate(['/login']);
    this.isMenuOpen = false; // Optional: close mobile menu on logout
  }
}