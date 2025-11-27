/**
 * User Model Interfaces
 * 
 * Purpose: TypeScript interfaces for user-related data structures
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

/**
 * User profile interface representing employee/manager profile information
 */
export interface UserProfile {
    /** Employee ID/username (numeric) */
    username: number;           
    /** Full name of the employee */
    employee_name: string | null;
    /** Name of the employee's manager */
    manager_name: string | null;
    /** User role: manager or employee */
    role: 'manager' | 'employee';
    /** Array of direct report employee IDs (for managers) */
    direct_reports: number[];
  }