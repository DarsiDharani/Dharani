/**
 * Competency Model Interfaces
 * 
 * Purpose: TypeScript interfaces for competency data structures
 * Used for CRUD operations and data transfer between frontend and backend
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

/**
 * User profile interface (duplicate of user.model.ts - consider consolidating)
 * Represents employee/manager profile information
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
    /** Array of direct report employee information (for managers) */
    direct_reports: DirectReport[];
  }

/**
 * Base competency interface containing all core competency fields
 * Used as base for create, update, and output interfaces
 */
export interface CompetencyBase {
    /** Employee ID */
    employee_empid: string;
    /** Employee name */
    employee_name:  string;
    /** Division the employee belongs to */
    division:       string;
    /** Role-specific competency area */
    role_specific_comp: string;
    /** Competency area */
    competency:     string;
    /** Specific skill name */
    skill:          string;
    /** Current expertise level (L1-L5) */
    current_expertise:  string;
    /** Target expertise level (L1-L5) */
    target_expertise: string;
  }
  
  /**
   * Interface for creating a new competency record
   * Extends CompetencyBase (no additional fields needed for creation)
   */
  export interface CompetencyCreate extends CompetencyBase {}
  
  /**
   * Interface for updating an existing competency record
   * Includes ID to identify which record to update
   */
  export interface CompetencyUpdate extends CompetencyBase {
    /** Unique identifier of the competency record to update */
    id: number;
  }
  
  /**
   * Interface for competency data returned from API
   * Includes ID assigned by the database
   */
  export interface CompetencyOut extends CompetencyBase {
    /** Unique identifier assigned by database */
    id: number;
  }
  
  /**
   * Interface combining user profile with their competencies
   * Used when fetching complete user data with all competencies
   */
  export interface CompetenciesWithUser {
    /** User profile information */
    user: UserProfile;
    /** Array of all competencies for this user */
    competencies: CompetencyOut[];
  }

  /**
   * Direct report interface
   * Represents an employee who reports to a manager
   */
  export interface DirectReport {
    /** Employee ID */
    employee_empid: string;
    /** Employee name */
    employee_name: string;
  }
