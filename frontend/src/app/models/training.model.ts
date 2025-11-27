/**
 * Training Model Interfaces
 * 
 * Purpose: TypeScript interfaces for training-related data structures
 * Used across the application for type safety and consistency
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

/**
 * Training detail interface representing a training session
 * Contains all information about a training including schedule, trainer, and content
 */
export interface TrainingDetail {
  /** Unique identifier for the training */
  id: number;
  /** Division this training belongs to */
  division?: string;
  /** Department this training belongs to */
  department?: string;
  /** Competency area this training covers */
  competency?: string;
  /** Specific skill this training addresses */
  skill?: string;
  /** Name/title of the training */
  training_name: string;
  /** Topics covered in the training */
  training_topics?: string;
  /** Prerequisites for attending this training */
  prerequisites?: string;
  /** Skill category level (L1, L2, L3, L4, L5) */
  skill_category?: string;
  /** Name of the trainer conducting the training */
  trainer_name: string;
  /** Trainer's email address */
  email?: string;
  /** Date when training is scheduled (ISO format string) */
  training_date?: string;
  /** Duration of the training (e.g., "2 hours", "1 day") */
  duration?: string;
  /** Time of day when training starts (e.g., "10:00 AM") */
  time?: string;
  /** Type of training: Online, Offline, or Hybrid */
  training_type?: string;
  /** Number of available seats */
  seats?: string;
  /** Details about the assessment/assignment for this training */
  assessment_details?: string;
  /** Type of assignment: personal (individual) or team (group) */
  assignmentType?: 'personal' | 'team';
  /** Employee ID if training is assigned to a specific person */
  assigned_to?: string;
}

/**
 * Training request interface for employee training requests
 * Used when employees request approval from managers to attend trainings
 */
export interface TrainingRequest {
  /** Unique identifier for the request */
  id: number;
  /** ID of the training being requested */
  training_id: number;
  /** Employee ID of the requester */
  employee_empid: string;
  /** Manager ID who needs to approve/reject */
  manager_empid: string;
  /** Date when the request was submitted (ISO format string) */
  request_date: string;
  /** Current status of the request */
  status: 'pending' | 'approved' | 'rejected';
  /** Optional notes from the manager */
  manager_notes?: string;
  /** Date when manager responded (ISO format string) */
  response_date?: string;
  /** Full training details object */
  training: TrainingDetail;
  /** Optional employee information */
  employee?: {
    /** Employee username/ID */
    username: string;
    /** Employee name */
    name?: string;
  };
}

/**
 * Calendar event interface for displaying trainings in calendar view
 * Simplified structure for calendar rendering
 */
export interface CalendarEvent {
  /** Date of the training event */
  date: Date;
  /** Title/name of the training */
  title: string;
  /** Name of the trainer */
  trainer: string;
}

