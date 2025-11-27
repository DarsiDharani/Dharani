/**
 * Assignment Model Interfaces
 * 
 * Purpose: TypeScript interfaces for assignment and feedback-related data structures
 * Used for training assignments, exams, and feedback forms
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

/**
 * Option for assignment questions (multiple choice, single choice)
 */
export interface QuestionOption {
  /** Text content of the option */
  text: string;
  /** Whether this option is the correct answer */
  isCorrect: boolean;
}

/**
 * Assignment question interface
 * Supports multiple question types: single-choice, multiple-choice, and text-input
 */
export interface AssignmentQuestion {
  /** The question text */
  text: string;
  /** Helper text providing additional instructions (e.g., "Please select at most 2 options.") */
  helperText: string;
  /** Type of question: single-choice, multiple-choice, or text-input */
  type: 'single-choice' | 'multiple-choice' | 'text-input';
  /** Array of answer options (not used for text-input type) */
  options: QuestionOption[];
}

/**
 * Complete assignment interface
 * Contains all questions and metadata for a training assignment
 */
export interface Assignment {
  /** ID of the training this assignment belongs to (null if not yet assigned) */
  trainingId: number | null;
  /** Title of the assignment */
  title: string;
  /** Description/instructions for the assignment */
  description: string;
  /** Array of questions in the assignment */
  questions: AssignmentQuestion[];
  /** ID of shared assignment if this is a shared assignment from trainer */
  sharedAssignmentId?: number;
}

/**
 * User's answer to an assignment question
 * Used when submitting assignment responses
 */
export interface UserAnswer {
  /** Index of the question being answered (0-based) */
  questionIndex: number;
  /** Type of question (single-choice, multiple-choice, text-input) */
  type: string;
  /** Array of selected option indices (for single/multiple choice) */
  selectedOptions: number[];
  /** Text answer (for text-input type questions) */
  textAnswer?: string;
}

/**
 * Result for a single question after grading
 * Used to display detailed feedback to users
 */
export interface QuestionResult {
  /** Index of the question (0-based) */
  questionIndex: number;
  /** Whether the user's answer was correct */
  isCorrect: boolean;
  /** Array of correct answer option indices */
  correctAnswers: number[];
  /** Array of user's selected answer option indices */
  userAnswers: number[];
  /** User's text answer (for text-input questions) */
  userTextAnswer?: string;
}

/**
 * Complete assignment result after submission
 * Contains overall score and detailed question results
 */
export interface AssignmentResult {
  /** Unique identifier for the result record */
  id: number;
  /** ID of the training this assignment belongs to */
  training_id: number;
  /** Final score (percentage) */
  score: number;
  /** Total number of questions */
  total_questions: number;
  /** Number of correctly answered questions */
  correct_answers: number;
  /** Detailed results for each question */
  question_results: QuestionResult[];
  /** Timestamp when assignment was submitted (ISO format string) */
  submitted_at: string;
}

/**
 * Feedback question interface
 * Used for post-training feedback forms
 */
export interface FeedbackQuestion {
  /** The feedback question text */
  text: string;
  /** Array of possible response options */
  options: string[];
  /** Whether this is a default/system question (vs. custom trainer question) */
  isDefault: boolean;
}

