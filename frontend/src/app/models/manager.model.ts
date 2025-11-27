/**
 * Manager Model Interfaces
 * 
 * Purpose: TypeScript interfaces for manager dashboard and team management
 * Contains interfaces for team members, competencies, and performance tracking
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Skill } from './skill.model';
import { AdditionalSkill } from './skill.model';

/**
 * Competency interface representing skill competency status
 * Used for both manager and team member skills
 */
export interface Competency {
  /** Name of the skill */
  skill: string;
  /** Competency area */
  competency: string;
  /** Current expertise level (L1-L5) */
  current_expertise: string;
  /** Target expertise level (L1-L5) */
  target_expertise: string;
  /** Status: Met (requirement satisfied), Gap (needs improvement), or Error */
  status: 'Met' | 'Gap' | 'Error';
}

/**
 * Team member interface representing an employee under a manager
 */
export interface TeamMember {
  /** Employee ID */
  id: string;
  /** Employee name */
  name: string;
  /** Array of core competencies/skills */
  skills: Competency[];
  /** Optional array of additional self-reported skills */
  additional_skills?: AdditionalSkill[];
}

/**
 * Manager data interface containing manager profile and team information
 * Main data structure for manager dashboard
 */
export interface ManagerData {
  /** Manager's name */
  name: string;
  /** Manager's role */
  role: string;
  /** Manager's employee ID */
  id: string;
  /** Manager's own skills/competencies */
  skills: Competency[];
  /** Array of team members under this manager */
  team: TeamMember[];
  /** Whether the manager is also a trainer */
  manager_is_trainer: boolean;
}

/**
 * Team assignment submission interface
 * Represents an assignment submission by a team member
 */
export interface TeamAssignmentSubmission {
  /** Unique identifier for the submission */
  id: number;
  /** ID of the training this assignment belongs to */
  training_id: number;
  /** Name of the training */
  training_name: string;
  /** Employee ID of the submitter */
  employee_empid: string;
  /** Name of the employee */
  employee_name: string;
  /** Score achieved (percentage) */
  score: number;
  /** Total number of questions */
  total_questions: number;
  /** Number of correct answers */
  correct_answers: number;
  /** Timestamp when submitted (ISO format string) */
  submitted_at: string;
  /** Whether feedback has been submitted for this training */
  has_feedback?: boolean;
  /** Number of feedback responses submitted */
  feedback_count?: number;
}

/**
 * Team feedback submission interface
 * Represents a feedback submission by a team member
 */
export interface TeamFeedbackSubmission {
  /** Unique identifier for the submission */
  id: number;
  /** ID of the training this feedback is for */
  training_id: number;
  /** Name of the training */
  training_name: string;
  /** Employee ID of the submitter */
  employee_empid: string;
  /** Name of the employee */
  employee_name: string;
  /** Array of feedback responses */
  responses: Array<{ questionIndex: number; questionText: string; selectedOption: string }>;
  /** Timestamp when submitted (ISO format string) */
  submitted_at: string;
}

/**
 * Manager performance feedback interface
 * Represents feedback given by manager about employee's training performance
 */
export interface ManagerPerformanceFeedback {
  /** Unique identifier for the feedback record */
  id: number;
  /** ID of the training this feedback is for */
  training_id: number;
  /** Name of the training */
  training_name: string;
  /** Employee ID being evaluated */
  employee_empid: string;
  /** Name of the employee */
  employee_name: string;
  /** Manager ID giving the feedback */
  manager_empid: string;
  /** Name of the manager */
  manager_name: string;
  /** Application of training in daily work score (optional, 1-5 scale) */
  application_of_training?: number;
  /** Quality of deliverables score (optional, 1-5 scale) */
  quality_of_deliverables?: number;
  /** Problem-solving capability score (optional, 1-5 scale) */
  problem_solving_capability?: number;
  /** Productivity & independence score (optional, 1-5 scale) */
  productivity_independence?: number;
  /** Process & compliance adherence score (optional, 1-5 scale) */
  process_compliance_adherence?: number;
  /** Areas where employee needs improvement */
  improvement_areas?: string;
  /** Employee's strengths */
  strengths?: string;
  /** Overall performance score (1-5 scale) */
  overall_performance: number;
  /** Additional comments from manager */
  additional_comments?: string;
  /** Timestamp when feedback was created (ISO format string) */
  created_at: string;
  /** Timestamp when feedback was last updated (ISO format string) */
  updated_at: string;
}

