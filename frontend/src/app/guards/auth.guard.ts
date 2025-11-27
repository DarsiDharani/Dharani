/**
 * Authentication Guard
 * 
 * Purpose: Protects routes that require user authentication
 * Features:
 * - Implements Angular CanActivate interface for route protection
 * - Checks if user is logged in before allowing route access
 * - Redirects unauthenticated users to login page
 * 
 * Usage:
 * Add to route configuration: { path: 'dashboard', component: DashboardComponent, canActivate: [AuthGuard] }
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Injectable } from '@angular/core';
import { CanActivate, Router } from '@angular/router';
import { AuthService } from '../services/auth.service';

@Injectable({
  providedIn: 'root',
})
export class AuthGuard implements CanActivate {
  /**
   * Guard constructor - injects required services
   * @param auth - AuthService for checking authentication status
   * @param router - Router for navigation to login page
   */
  constructor(private auth: AuthService, private router: Router) {}

  /**
   * Determines if a route can be activated
   * Checks if user is authenticated, redirects to login if not
   * @returns true if user is logged in, false otherwise (triggers redirect)
   */
  canActivate(): boolean {
    if (!this.auth.isLoggedIn()) {
      this.router.navigate(['/login']);
      return false;
    }
    return true;
  }
}
