/**
 * Authentication Service
 * 
 * Purpose: Manages user authentication state and session data
 * Features:
 * - Store and retrieve authentication token
 * - Manage user role and username
 * - Check login status
 * - Clear authentication data on logout
 * 
 * Storage Keys:
 * - 'access_token': JWT token for API authentication
 * - 'user_role': User's role (manager/employee)
 * - 'username': User's employee ID/username
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  /**
   * Service constructor
   */
  constructor() {}

  /**
   * Stores authentication data in localStorage
   * Called after successful login
   * @param token - JWT access token from API
   * @param role - User role ('manager' or 'employee')
   * @param username - Employee ID/username
   */
  login(token: string, role: string, username: string) {
    localStorage.setItem('access_token', token);
    localStorage.setItem('user_role', role);
    localStorage.setItem('username', username);
  }

  /**
   * Clears authentication data from localStorage
   * Only removes auth-related items, preserves other application data
   * Called on logout or token expiration
   */
  logout() {
    // Only clear authentication-related data, preserve other application data
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_role');
    localStorage.removeItem('username');
    // Note: We intentionally keep other localStorage items that might be needed
    // for application state or user preferences
  }

  /**
   * Checks if user is currently logged in
   * @returns true if access token exists, false otherwise
   */
  isLoggedIn(): boolean {
    return !!this.getToken();
  }

  /**
   * Retrieves the user's role from localStorage
   * @returns User role string ('manager' or 'employee') or null if not set
   */
  getRole(): string | null {
    return localStorage.getItem('user_role');
  }

  /**
   * Retrieves the employee ID from localStorage
   * Note: Employee ID is stored as 'username' in localStorage
   * @returns Employee ID string or null if not set
   */
  getEmpId(): string | null {
    return localStorage.getItem('username');
  }

  /**
   * Retrieves the username from localStorage
   * @returns Username string or null if not set
   */
  getUsername(): string | null {
    return localStorage.getItem('username');
  }

  /**
   * Retrieves the access token from localStorage
   * @returns JWT token string or null if not set
   */
  getToken(): string | null {
    return localStorage.getItem('access_token');
  }
}