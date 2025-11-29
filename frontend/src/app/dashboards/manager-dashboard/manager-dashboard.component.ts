/**
 * Manager Dashboard Component
 * 
 * Purpose: Main dashboard for managers to manage team skills, assign trainings, and track performance
 * 
 * Features:
 * - Team Management:
 *   - View personal skills and team member skills
 *   - Track team skill gaps and progress
 *   - Edit team member skill levels
 *   - View team competency matrices
 * 
 * - Training Management:
 *   - Assign trainings to team members (bulk assignment support)
 *   - View all available trainings
 *   - Manage training requests from team members (approve/reject)
 *   - View training calendar
 * 
 * - Performance Tracking:
 *   - View team assignment submissions and scores
 *   - View team feedback submissions
 *   - Provide performance feedback to team members
 *   - Track team training completion rates
 * 
 * - Dashboard Metrics:
 *   - Team size and skill statistics
 *   - Top skill gaps across team
 *   - Assigned trainings count
 *   - Personal vs. team view toggle
 * 
 * Key Sections:
 * 1. Dashboard Tab: Overview with metrics and team statistics
 * 2. My Skills Tab: Manager's personal skills (core and additional)
 * 3. Team Skills Tab: View and manage team member skills
 * 4. Training Catalog Tab: Browse all available trainings
 * 5. Assign Training Tab: Assign trainings to team members
 * 6. Training Requests Tab: Approve/reject team member training requests
 * 7. Team Assignments Tab: View team assignment submissions
 * 8. Team Feedback Tab: View team feedback submissions
 * 9. Performance Feedback Tab: Provide feedback to team members
 * 
 * State Management:
 * - Manages selected trainings and team members for bulk operations
 * - Tracks duplicate assignments to prevent conflicts
 * - Uses filters for various views
 * 
 * API Integration:
 * - Uses ApiService for centralized endpoint management
 * - Implements proper error handling with ToastService
 * - Handles authentication via AuthService
 * - Uses forkJoin for parallel API calls
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit, ElementRef, QueryList, AfterViewInit, ViewChildren } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { ApiService } from '../../services/api.service';
import { trigger, style, animate, transition, query, stagger } from '@angular/animations';
import { ToastService, ToastMessage } from '../../services/toast.service';
import { NotificationService } from '../../services/notification.service';
import { forkJoin, of } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { TrainingDetail, TrainingRequest, CalendarEvent } from '../../models/training.model';
import { Assignment, AssignmentQuestion, QuestionOption, FeedbackQuestion } from '../../models/assignment.model';

interface Competency {
  skill: string;
  competency: string;
  current_expertise: string;
  target_expertise: string;
  status: 'Met' | 'Gap' | 'Error';
}

interface AdditionalSkill {
  id: number;
  skill_name: string;
  skill_level: string;
  skill_category: string;
  description?: string;
  created_at?: string;
}

interface TeamMember {
  id: string;
  name: string;
  skills: Competency[];
  additional_skills?: AdditionalSkill[];
}

interface ManagerData {
  name: string;
  role: string;
  id: string;
  skills: Competency[];
  team: TeamMember[];
  manager_is_trainer: boolean;
}

// TrainingDetail and TrainingRequest are now imported from models/training.model.ts

interface TeamAssignmentSubmission {
    id: number;
    training_id: number;
    training_name: string;
    employee_empid: string;
    employee_name: string;
    score: number;
    total_questions: number;
    correct_answers: number;
    submitted_at: string;
    has_feedback?: boolean;
    feedback_count?: number;
}

interface TeamFeedbackSubmission {
    id: number;
    training_id: number;
    training_name: string;
    employee_empid: string;
    employee_name: string;
    responses: Array<{ questionIndex: number; questionText: string; selectedOption: string }>;
    submitted_at: string;
    has_feedback?: boolean;
    feedback_count?: number;
}

interface ManagerPerformanceFeedback {
    id: number;
    training_id: number;
    training_name: string;
    employee_empid: string;
    employee_name: string;
    manager_empid: string;
    manager_name: string;
    application_of_training?: number;
    quality_of_deliverables?: number;
    problem_solving_capability?: number;
    productivity_independence?: number;
    process_compliance_adherence?: number;
    improvement_areas?: string;
    strengths?: string;
    overall_performance: number;
    additional_comments?: string;
    created_at: string;
    updated_at: string;
    updateNumber?: number;  // Added for feedback history display
    totalUpdates?: number;  // Added for feedback history display
    skill_category?: string;  // Added for skill level grouping (L1-L5)
}

// CalendarEvent, Assignment, AssignmentQuestion, QuestionOption, and FeedbackQuestion 
// are now imported from models

type LevelBlock = { level: number; items: string[] };
type Section = { title: string; subtitle?: string; levels: LevelBlock[] };

@Component({
  selector: 'app-manager-dashboard',
  templateUrl: './manager-dashboard.component.html',
  styleUrls: ['./manager-dashboard.component.css'],
  animations: [
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0 }),
        animate('500ms ease-in', style({ opacity: 1 }))
      ]),
      transition(':leave', [
        animate('500ms ease-out', style({ opacity: 0 }))
      ])
    ]),
    trigger('slideFadeIn', [
        transition(':enter', [
            style({ opacity: 0, transform: 'translateY(-20px)' }),
            animate('500ms ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
        ]),
        transition(':leave', [
            animate('500ms ease-in', style({ opacity: 0, transform: 'translateY(-20px)' }))
        ])
    ]),
    trigger('modalScale', [
      transition(':enter', [
        style({ opacity: 0, transform: 'scale(0.95)' }),
        animate('300ms ease-out', style({ opacity: 1, transform: 'scale(1)' }))
      ]),
      transition(':leave', [
        animate('300ms ease-in', style({ opacity: 0, transform: 'scale(0.95)' }))
      ])
    ]),
    trigger('listStagger', [
      transition('* <=> *', [
        query(':enter', [
          style({ opacity: 0, transform: 'translateY(20px)' }),
          stagger('120ms', animate('600ms ease-out', style({ opacity: 1, transform: 'translateY(0)' })))
        ], { optional: true })
      ])
    ]),
    trigger('bouncyScale', [
      transition(':enter', [
        style({ transform: 'scale(0.5)', opacity: 0 }),
        animate('700ms cubic-bezier(0.68, -0.55, 0.27, 1.55)', style({ transform: 'scale(1)', opacity: 1 }))
      ])
    ])
  ]
})
export class ManagerDashboardComponent implements OnInit, AfterViewInit {
  @ViewChildren('animatedElement') animatedElements!: QueryList<ElementRef>;
  private observer!: IntersectionObserver;

  activeTab: string = 'dashboard';
  manager: ManagerData | null = null;
  managerDisplayName: string = '';
  managerIsTrainer: boolean = false;
  isLoading = true;
  errorMessage: string = '';
  successMessage: string = '';
  totalTeamMembers: number = 0;
  teamSkillsMet: number = 0;
  teamSkillsGap: number = 0;
  topSkillGaps: any[] = [];
  assignedTrainingsCount: number = 0; // New metric!
  
  // Dashboard view toggle
  dashboardView: 'personal' | 'team' = 'personal';
  pinnedItems: string[] = []; // For pin-to-pin feature
  
  selectedTeamMember: TeamMember | null = null;

  editingSkill: { memberId: string, skillIndex: number } | null = null;
  editSkillData: { current_expertise: string, target_expertise: string } = { current_expertise: '', target_expertise: '' };

  selectedTrainingIds: number[] = [];
  selectedMemberIds: string[] = [];

  assignTrainingSearch: string = '';
  assignMemberSearch: string = '';

  // Success modal properties
  showAssignmentSuccessModal: boolean = false;
  assignmentSuccessData: { trainingNames: string[]; memberNames: string[]; totalAssignments: number } | null = null;
  isAssigningTraining: boolean = false;
  
  // Duplicate assignments modal
  showDuplicateModal: boolean = false;
  duplicateAssignments: { training: string; member: string }[] = [];
  pendingValidAssignments: { trainingId: number; memberId: string; trainingName: string; memberName: string }[] = [];
  
  // Existing assignments for duplicate checking (format: "trainingId_employeeId")
  existingAssignments: Set<string> = new Set();

  mySkillsStatusFilter: 'All' | 'Met' | 'Gap' = 'All';
  mySkillsSkillFilter: 'All' | string = 'All';
  mySkillsSearch: string = '';
  mySkillsView: 'core' | 'additional' = 'core'; // New property for the toggle UI
  teamSkillsStatusFilter: 'All' | 'Met' | 'Gap' = 'All';
  teamSkillsSkillFilter: 'All' | string = 'All';
  teamMemberNameFilter: 'All' | string = 'All';
  teamCompetencyFilter: 'All' | string = 'All';
  teamSkillsCurrentLevelFilter: 'All' | string = 'All';

  uniqueMySkills: string[] = [];
  uniqueTeamSkills: string[] = [];
  uniqueTeamMembers: string[] = [];
  uniqueCompetencies: string[] = [];
  uniqueCurrentLevels: string[] = [];

  skillLevels: string[] = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'];
  skillLevelsForFilter: string[] = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'];
  
  skillCategoryLevels: string[] = ['L1', 'L2', 'L3', 'L4', 'L5'];
  skillNames: string[] = [];

  additionalSkills: any[] = [];
  newSkill = {
    name: '',
    level: 'Beginner',
    category: 'Technical',
    description: ''
  };
  showAddSkillForm: boolean = false;
  editingSkillId: number | null = null;
  additionalSkillLevels = ['Beginner', 'Intermediate', 'Advanced', 'Expert'];
  skillCategories = ['Technical', 'Soft Skills', 'Leadership', 'Communication', 'Project Management', 'Other'];
  
  showScheduleTrainingModal = false;
  trainingCatalog: TrainingDetail[] = [];
  allTrainings: TrainingDetail[] = [];
  assignedTrainings: TrainingDetail[] = [];
  teamAssignedTrainings: TrainingDetail[] = [];
  pendingRequests: TrainingRequest[] = [];
  catalogSearch: string = '';
  catalogTypeFilter: string = 'All';
  catalogCategoryFilter: string = 'All';
  
  // --- Assigned Trainings Filters ---
  assignedSearch: string = '';
  assignedSkillFilter: string = 'All';
  assignedLevelFilter: string = 'All';
  assignedDateFilter: string = '';
  assignedTrainingsView: 'list' | 'calendar' = 'list';
  trainingCatalogView: 'list' | 'calendar' = 'list';

  // --- Calendar & Dashboard Metrics ---
  allTrainingsCalendarEvents: CalendarEvent[] = [];
  assignedTrainingsCalendarEvents: CalendarEvent[] = [];
  currentDate: Date = new Date();
  calendarDays: (Date | null)[] = [];
  calendarMonth: string = '';
  calendarYear: number = 2025;

  // --- Trainer Zone Properties ---
  isTrainer: boolean = false;
  trainerZoneView: 'overview' | 'assignmentForm' | 'feedbackForm' = 'overview';
  sharedAssignments: Map<number, boolean> = new Map(); // Track which trainings have assignments shared
  sharedFeedback: Map<number, boolean> = new Map(); // Track which trainings have feedback shared
  assignmentSharedBy: Map<number, string> = new Map(); // Track who shared the assignment
  feedbackSharedBy: Map<number, string> = new Map(); // Track who shared the feedback
  trainingCandidates: Map<number, {employee_empid: string, employee_name: string, attended: boolean}[]> = new Map(); // Store candidates for each training
  showAttendanceModal: boolean = false;
  selectedTrainingForAttendance: number | null = null;
  attendanceCandidates: {employee_empid: string, employee_name: string, attended: boolean}[] = [];
  isMarkingAttendance: boolean = false;
  private _myTrainingsCache: TrainingDetail[] = [];
  private _myTrainingsCacheKey: string = '';
  
  // File management properties
  questionFilesUploaded: Map<number, boolean> = new Map(); // Track which trainings have question files
  trainerSolutions: Map<number, any[]> = new Map(); // Store solution files for trainers to view
  showSolutionsModal: boolean = false;
  selectedTrainingForSolutions: number | null = null;
  solutionsList: any[] = [];
  isLoadingSolutions: boolean = false;
  
  // Assignment and Feedback Forms
  newAssignment: Assignment = {
    trainingId: null,
    title: '',
    description: '',
    questions: []
  };
  
  defaultFeedbackQuestions: FeedbackQuestion[] = [
    { text: "How would you rate your overall experience with this training?", options: ['Excellent', 'Good', 'Average', 'Fair', 'Poor'], isDefault: true },
    { text: "Was the content relevant and applicable to your role?", options: ['Yes', 'No', 'Partially'], isDefault: true },
    { text: "Was the material presented in a clear and understandable way?", options: ['Yes', 'No', 'Somewhat'], isDefault: true },
    { text: "Did the training meet your expectations?", options: ['Yes', 'No', 'Partially'], isDefault: true },
    { text: "Was the depth of the content appropriate?", options: ['Appropriate', 'Too basic', 'Too advanced'], isDefault: true },
    { text: "Was the trainer able to explain concepts clearly?", options: ['Yes', 'No', 'Somewhat'], isDefault: true },
    { text: "Did the trainer engage participants effectively?", options: ['Yes', 'No', 'Somewhat'], isDefault: true },
    { text: "Will this training improve your day-to-day job performance?", options: ['Yes', 'No', 'Maybe'], isDefault: true },
    { text: "Was the pace of the training comfortable?", options: ['Comfortable', 'Too fast', 'Too slow'], isDefault: true },
    { text: "Were the training materials/resources useful?", options: ['Yes', 'No', 'Somewhat'], isDefault: true }
  ];
  newFeedback = {
    trainingId: null as number | null,
    customQuestions: [] as FeedbackQuestion[]
  };
  
  newTraining = {
    division: '',
    department: '',
    competency: '',
    skill: '',
    training_name: '',
    training_topics: '',
    prerequisites: '',
    skill_category: 'L1',
    trainer_name: '',
    email: '',
    training_date: '',
    duration: '',
    time: '',
    training_type: 'Online',
    seats: '',
    assessment_details: ''
  };

  // Modal properties
  showDetailModal = false;
  modalTitle = '';
  modalData: any[] = [];
  modalDataType: 'members' | 'skills' | null = null;

  // Team submissions
  teamAssignmentSubmissions: TeamAssignmentSubmission[] = [];
  teamFeedbackSubmissions: TeamFeedbackSubmission[] = [];
  isLoadingSubmissions: boolean = false;

  // Performance feedback
  showFeedbackModal: boolean = false;
  selectedSubmissionForFeedback: { type: 'assignment' | 'feedback'; submission: TeamAssignmentSubmission | TeamFeedbackSubmission } | null = null;
  performanceFeedback = {
    training_id: 0,
    employee_empid: '',
    application_of_training: null as number | null,
    quality_of_deliverables: null as number | null,
    problem_solving_capability: null as number | null,
    productivity_independence: null as number | null,
    process_compliance_adherence: null as number | null,
    improvement_areas: '',
    strengths: '',
    overall_performance: 3,
    additional_comments: ''
  };
  isSubmittingFeedback: boolean = false;
  existingFeedback: ManagerPerformanceFeedback | null = null;
  feedbackHistory: ManagerPerformanceFeedback[] = [];
  feedbackHistoryByType: Map<string, ManagerPerformanceFeedback[]> = new Map();
  isLoadingFeedbackHistory: boolean = false;
  showFeedbackHistoryModal: boolean = false;
  selectedSkillTypeForHistory: string = '';
  selectedTrainingNameForHistory: string = '';
  selectedEmployeeNameForHistory: string = '';
  feedbackHistoryList: ManagerPerformanceFeedback[] = [];
  showSuccessPopup: boolean = false;
  successPopupMessage: string = '';
  
  // Attendance success popup
  showAttendanceSuccessPopup: boolean = false;
  attendanceSuccessData: {
    trainingName: string;
    attendedCount: number;
    absentCount: number;
    totalCount: number;
    attendedNames: string;
  } | null = null;

  // Final Evaluation
  showFinalEvaluationModal: boolean = false;
  employeeSkillsForEvaluation: Array<{
    skill: string;
    current_expertise: string;
    target_expertise: string;
    targetAchieved: boolean;
  }> = [];
  isLoadingEmployeeSkills: boolean = false;
  isSubmittingEvaluation: boolean = false;
  isManagerSatisfied: boolean | null = null;
  isReassigningTraining: boolean = false;

  private readonly API_ENDPOINT = '/data/manager/dashboard';

  levelsSearch = '';
  selectedSkill = '';
  public expandedLevels = new Set<string>();
  public expandedSkill: string | null = null;
  levelHeaders = [
    { num: 1, title: 'Beginner' },
    { num: 2, title: 'Basic' },
    { num: 3, title: 'Intermediate' },
    { num: 4, title: 'Advanced' },
    { num: 5, title: 'Expert' }
  ];
  sections: Section[] = [
    {
      title: 'EXAM',
      levels: [
        { level: 1, items: ['Launch EXAM', 'Test execution', 'Exporting reports'] },
        { level: 2, items: ['Implement test cases', 'Create collections', 'EXAM configuration', 'DOORS synchronization'] },
        { level: 3, items: ['Create short names', 'NeKeDa reporting', 'Debugging in EXAM'] },
        { level: 4, items: ['Implement libraries & common sequences', 'Know-how on libraries (1—4)', 'Create baselines', 'Release configuration', 'Update variable mapping', 'System configurations'] },
        { level: 5, items: ['EXAM administration', 'Model domain configuration', 'Set up new project', 'Groovy scripting'] }
      ]
    },
    {
      title: 'Softcar',
      levels: [
        { level: 1, items: ['Launch Softcar', 'Artifacts in Softcar'] },
        { level: 2, items: ['Blockboard & calibration variables', 'Add A2L variables', 'Logging'] },
        { level: 3, items: ['Debugging', 'Error simulation', 'CAN message error simulations', 'Script execution in Softcar'] },
        { level: 4, items: ['Create layouts', 'Trigger files', 'Startup script', 'Softcar scripting'] },
        { level: 5, items: ['Plant model creation', 'CAN configuration', 'DLL files'] }
      ]
    },
    {
      title: 'Python',
      subtitle: 'Foundational → application automation',
      levels: [
        { level: 1, items: ['Install packages', 'Syntax, data types, operators', 'Reserved keywords', 'Input/Output'] },
        { level: 2, items: ['Loops (for/while)', 'try/except', 'Strings', 'Lists/Dict/Sets methods', 'break/continue'] },
        { level: 3, items: ['Functions (incl. lambda, *args/**kwargs)', 'File handling', 'List comprehensions', 'Intro to classes & objects'] },
        { level: 4, items: ['Inheritance, encapsulation, polymorphism', 'Static/class methods', 'json', 'pip packages', 'Debugging with pdb & argparse'] },
        { level: 5, items: ['API requests', 'App development', 'Task automation with libs', 'Decorators & generators', 'Excel data ops, pickling'] }
      ]
    },
    {
      title: 'C++ (CPP)',
      subtitle: 'Skill matrix by domain',
      levels: [
        { level: 1, items: ['Core language: variables, loops, conditionals, functions', 'Memory: stack, basic pointers', 'OOP/Templates: basic class/struct, simple encapsulation', 'Std libs: I/O streams, arrays', 'Concurrency: none/very basic', 'HW interaction: none', 'Build: single-file or simple compile', 'Debugging: print-based', 'Architecture: simple procedural'] },
        { level: 2, items: ['Core: classes, inheritance, function overloading', 'Memory: dynamic allocation, manual new/delete', 'OOP: full OOP, virtual functions, basic templates', 'Std libs: STL containers, iterators, namespaces', 'Concurrency: std::thread, mutexes, condition variables, basics', 'HW: UART/SPI/I2C basics, polling/interrupt basics', 'Build: CMake/make projects', 'Debugging: IDE debuggers, tracing', 'Architecture: modular design, class hierarchies'] },
        { level: 3, items: ['Core: smart pointers, templates, lambda, move semantics', 'Memory: unique/shared pointers, RAII', 'OOP: template classes, function templates, partial specialization', 'Std libs: STL algorithms, functional (bind, function), C++17/20 features', 'Concurrency: thread pools, atomics, lock-free queues', 'HW: protocol stacks, parsing/filtering sensor data, bootloader integration', 'Build: cross-compilation, linker script editing, startup code', 'Debugging: HW breakpoints, test-driven development', 'Architecture: HAL & layered driver-service-app'] },
        { level: 4, items: ['Core: advanced metaprogramming, constexpr, concepts, compile-time programming', 'Memory: custom allocators, memory pools, cache-aware structures, linker scripts', 'OOP: CRTP, SFINAE, concepts, policy-based design', 'Std libs: custom allocators/customization, deep C++20/23 (coroutines, ranges)', 'Concurrency: RTOS integration, scheduling, context switching, real-time tuning', 'HW: firmware architecture, power optimization, watchdogs, interrupt prioritization', 'Build: advanced CMake, memory maps, flash/ROM segmentation, compiler flags', 'Debugging: JTAG/SWD, oscilloscopes, profilers, automated pipelines', 'Architecture: full firmware, distributed systems, safety compliance (MISRA/ISO 26262)'] }
      ]
    },
    {
      title: 'Axivion',
      levels: [
        { level: 1, items: ['Batch run', 'Review reports', 'Fix issues'] },
        { level: 2, items: ['Tool configuration', 'Refine issues (false positives, severity, incremental analysis, trace bugs)'] },
        { level: 3, items: ['Define architecture model (layered, client-server)', 'Verify dependencies', 'Detect cycles/layer violations/illegal access', 'Issue baselines to isolate new violations'] },
        { level: 4, items: ['CI/CD report generation (Git/Jenkins)', 'Compliance (MISRA/AUTOSAR/ISO26262 traceability)'] },
        { level: 5, items: ['Scripting: custom rules (naming, complexity limits)', 'Combine with other tools'] }
      ]
    },
    {
      title: 'MATLAB',
      levels: [
        { level: 1, items: ['Launch MATLAB'] },
        { level: 2, items: ['Configuration & inputs', 'Variable handling', 'Execution'] },
        { level: 3, items: ['Debugging (Simulink)', 'Error simulation', 'M-script execution'] },
        { level: 4, items: ['M-scripting', 'Stateflow debugging', 'Create S-Function'] },
        { level: 5, items: ['Library creation', 'Module implementation'] }
      ]
    },
    {
      title: 'DOORS',
      levels: [
        { level: 1, items: ['UI navigation (modules, views, folders)', 'Toolbar/menus/commands basics'] },
        { level: 2, items: ['Create/edit/manage requirements', 'Link requirements for traceability', 'Use attributes for categorize/filter'] },
        { level: 3, items: ['DB setup/maintenance', 'Import/export data', 'Manage users & permissions'] },
        { level: 4, items: ['Customize views/layouts', 'DXL scripting & automation', 'Reports for coverage/traceability'] },
        { level: 5, items: ['Built-in analysis for gaps/inconsistencies', 'Integrations (IBM Rational, MS Office, etc.)'] }
      ]
    },
    {
      title: 'Azure DevOps',
      levels: [
        { level: 1, items: ['Access & overview of Pipelines and advantages'] },
        { level: 2, items: ['Run pipelines', 'Dashboard analysis', 'Produced/consumed artifacts'] },
        { level: 3, items: ['Debug pipeline errors', 'Know Azure services: Boards, Repos, Pipeline Library'] },
        { level: 4, items: ['Agents, Pools, Stages, Jobs, Builds, Variables', 'Variable groups, PAT, Resources'] },
        { level: 5, items: ['Create pipelines with YAML', 'Full pipeline creation & Azure dashboard administration'] }
      ]
    },
    {
      title: 'Smart Git',
      levels: [
        { level: 1, items: ['Can open Smart Git and perform basic operations like viewing repositories and navigating the tool interface', 'Understand the concept of git version control', 'Can clone a repository', 'Basic knowledge of Git concepts (add, stage, stash, commit, fetch, push, pull) but lacks deeper understanding'] },
        { level: 2, items: ['Branch management', 'Comfortable using Smart Git for basic Git workflows like creating and switching branches, merging, and resolving simple merge conflicts', 'Should know about .git file configuration', 'Good Hands on Git operations (commit, push, fetch, pull, pull requests..etc)', 'Has a basic understanding of how Git works (branching, commits, merges)', 'Understands the concept of merge conflicts and can resolve them with some help', 'Able to know the changes in the commit history itself and understand differences between versions'] },
        { level: 3, items: ['Should expert in branch rebasing b/w multiple task branches or main branch.etc', 'Advanced features like rebase, cherry-pick, or interactive rebasing', 'Understands and can explain how Git handles data (how commits work, SHA-1 hashes, etc.)'] },
        { level: 4, items: ['Can diagnose and resolve issues that arise in project (e.g., complex merge conflicts, history rewrites, etc.)', 'Deep understanding of Git internals, workflows, and advanced features like Git hooks, submodules, and CI/CD integration'] },
        { level: 5, items: ['Expert at troubleshooting and fixing complex Git issues, including history rewrites, reflog, and rebasing across multiple branches', 'Can mentor others, guide teams in setting up version control, and resolve any version control-related conflicts'] }
      ]
    },
    {
      title: 'Integrity',
      levels: [
        { level: 1, items: ['Configuration of Tool', 'Changing status of tasks', 'Attaching reports', 'Updating fields properly'] },
        { level: 2, items: ['Creating filters', 'Creating change requests', 'Spawns to change request', 'Delivery, Build', 'Review checklist creation', 'Creating member links', 'Creating sandboxes'] },
        { level: 3, items: ['Check-in and checkout of documents', 'Performing reviews', 'Tracing changes from Integrity to Source code'] },
        { level: 4, items: ['Generating reports to track progress and identify issues', 'Customizing reports to meet specific stakeholder needs'] },
        { level: 5, items: ['Managing user roles and permissions to ensure secure collaboration', 'Integrating with other PTC products and third party tools like Jira and Microsoft Teams'] }
      ]
    }
  ];

  constructor(
    private http: HttpClient,
    private router: Router,
    private authService: AuthService,
    private toastService: ToastService,
    private apiService: ApiService,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    this.loadPinnedItems();
    this.fetchDashboardData();
    this.fetchTrainingCatalog();
    this.fetchScheduledTrainings();
    this.fetchAssignedTrainings();
    this.fetchAssignedTrainingsCount();
    this.fetchTeamAssignedTrainings();
    this.fetchPendingRequests();
    this.fetchTeamSubmissions();
    // Initialize notifications
    this.notificationService.initialize();
  }

  fetchAssignedTrainingsCount(): void {
    // This is a placeholder. In a real app, you would make an API call here.
    // For now, we'll simulate a count.
    this.assignedTrainingsCount = 12; 
  }

  ngAfterViewInit(): void {
    this.observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('fade-in-up');
          this.observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1
    });

    this.animatedElements.changes.subscribe((comps: QueryList<ElementRef>) => {
      comps.forEach(el => this.observer.observe(el.nativeElement));
    });
  }

  selectTab(tabName: string): void {
    if (tabName === 'trainingCatalog' || tabName === 'assignTraining' || tabName === 'trainerZone') {
        this.fetchTrainingCatalog();
    }
    if (tabName === 'trainerZone') {
      // Refresh scheduled trainings when switching to Trainer Zone to show latest sessions
      this.fetchScheduledTrainings();
    }
    if (tabName === 'assignedTrainings') {
        this.fetchAssignedTrainings();
    }
    this.selectedTeamMember = null;
    this.mySkillsStatusFilter = 'All';
    this.mySkillsSkillFilter = 'All';
    this.teamSkillsStatusFilter = 'All';
    this.teamSkillsSkillFilter = 'All';
    this.teamMemberNameFilter = 'All';
    this.teamCompetencyFilter = 'All';
    this.teamSkillsCurrentLevelFilter = 'All';
    this.activeTab = tabName;
    setTimeout(() => {
        this.animatedElements.forEach(el => this.observer.observe(el.nativeElement));
    }, 0);
  }

  // Dashboard view toggle methods
  // ============================
  // === CORRECTED THIS BLOCK ===
  // ============================
  toggleDashboardView(): void {
    // The call to fetchTeamAssignedTrainings() was removed from here.
    // It's already called in ngOnInit(), so the data is loaded once when the component
    // initializes, which is more efficient.
    this.dashboardView = this.dashboardView === 'personal' ? 'team' : 'personal';
    if (this.dashboardView === 'team') {
      this.fetchTeamSubmissions(); // Refresh submissions when switching to team view
    }
  }

  // Pin-to-pin feature methods
  togglePin(item: string): void {
    const index = this.pinnedItems.indexOf(item);
    if (index > -1) {
      this.pinnedItems.splice(index, 1);
    } else {
      this.pinnedItems.push(item);
    }
    // Save to localStorage for persistence
    localStorage.setItem('managerPinnedItems', JSON.stringify(this.pinnedItems));
  }

  isPinned(item: string): boolean {
    return this.pinnedItems.includes(item);
  }

  loadPinnedItems(): void {
    const saved = localStorage.getItem('managerPinnedItems');
    if (saved) {
      this.pinnedItems = JSON.parse(saved);
    }
  }

  // Team dashboard specific methods
  getTeamSkillGaps(): any[] {
    if (!this.manager) return [];
    const allTeamSkills = this.manager.team.flatMap(member => member.skills);
    const skillGapCount: { [key: string]: number } = {};
    allTeamSkills.forEach(skill => {
      if (skill.status === 'Gap') {
        skillGapCount[skill.skill] = (skillGapCount[skill.skill] || 0) + 1;
      }
    });
    return Object.entries(skillGapCount)
      .map(([skill, count]) => ({ skill, count }))
      .sort((a, b) => b.count - a.count);
  }

  getTeamMembersWithGaps(): TeamMember[] {
    if (!this.manager) return [];
    return this.manager.team.filter(member => 
      member.skills.some(skill => skill.status === 'Gap')
    );
  }

  getTeamProgressByMember(): { member: TeamMember; progress: number }[] {
    if (!this.manager) return [];
    return this.manager.team.map(member => ({
      member,
      progress: this.calculateProgress(member.skills)
    })).sort((a, b) => b.progress - a.progress);
  }

  getUpcomingTeamTrainings(): TrainingDetail[] {
    // This would typically come from an API call for team assigned trainings
    return this.assignedTrainings.filter(t => 
      t.training_date && new Date(t.training_date) >= new Date()
    ).slice(0, 5);
  }

  // Get trainings assigned by the manager to team members
  getTeamAssignedTrainings(): TrainingDetail[] {
    // This will be populated from the team assigned trainings API call
    return this.teamAssignedTrainings || [];
  }

  // Get the name of the team member for a given employee ID
  getAssignedMemberName(employeeId: string | undefined): string {
    if (!employeeId || !this.manager || !this.manager.team) return employeeId || 'Unknown';
    const member = this.manager.team.find(m => m.id === employeeId);
    return member ? member.name : employeeId;
  }

  // Personal dashboard methods (same as engineer dashboard)
  getPersonalProgressPercentage(): number {
    if (!this.manager?.skills?.length) return 0;
    const totalSkills = this.manager.skills.length;
    const metSkills = this.getMySkillsMetCount();
    return totalSkills > 0 ? Math.round((metSkills / totalSkills) * 100) : 0;
  }

  getUpcomingPersonalTrainings(): TrainingDetail[] {
    if (!this.assignedTrainings || this.assignedTrainings.length === 0) {
      return [];
    }
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    return this.assignedTrainings
      .filter(t => t.training_date && new Date(t.training_date) >= today)
      .sort((a, b) => {
        return new Date(a.training_date!).getTime() - new Date(b.training_date!).getTime();
      });
  }

  highlightUpcomingTrainings(): void {
    const element = document.getElementById('upcoming-trainings-section');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.style.transition = 'box-shadow 0.5s ease-in-out';
      element.style.boxShadow = '0 0 0 4px #38bdf8, 0 0 15px #0ea5e9';
      setTimeout(() => {
        element.style.boxShadow = 'none';
      }, 2500);
    }
  }

  getTrainingCardIcon(skill?: string | null): string {
    if (!skill) return 'fa-solid fa-laptop-code';
    const skillLower = skill.toLowerCase();
    if (skillLower.includes('softcar')) return 'fa-solid fa-car';
    if (skillLower.includes('integrity')) return 'fa-solid fa-shield-halved';
    if (skillLower.includes('exam')) return 'fa-solid fa-microscope';
    if (skillLower.includes('cpp') || skillLower.includes('c++')) return 'fa-solid fa-code';
    if (skillLower.includes('python')) return 'fa-brands fa-python';
    if (skillLower.includes('matlab')) return 'fa-solid fa-chart-line';
    if (skillLower.includes('doors')) return 'fa-solid fa-door-open';
    if (skillLower.includes('azure')) return 'fa-brands fa-microsoft';
    if (skillLower.includes('git')) return 'fa-brands fa-git-alt';
    if (skillLower.includes('axivion')) return 'fa-solid fa-search';
    return 'fa-solid fa-laptop-code';
  }

  // Fetch team assigned trainings
  fetchTeamAssignedTrainings(): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.http.get<{training_id: number; employee_empid: string}[]>(this.apiService.managerTeamAssignmentsUrl, { headers }).subscribe({
      next: (response) => {
        console.log('Team assigned trainings loaded:', response);
        // Store existing assignments in a Set for quick duplicate checking
        this.existingAssignments.clear();
        (response || []).forEach((assignment: {training_id: number; employee_empid: string}) => {
          const key = `${assignment.training_id}_${assignment.employee_empid}`;
          this.existingAssignments.add(key);
        });
        // Also store full training details if needed
        this.teamAssignedTrainings = [];
      },
      error: (err) => {
        console.error('Failed to fetch team assigned trainings:', err);
        this.existingAssignments.clear();
        this.teamAssignedTrainings = [];
      }
    });
  }

  // Fetch team assignment and feedback submissions
  fetchTeamSubmissions(): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.isLoadingSubmissions = true;
    
    // Fetch both assignment and feedback submissions
    forkJoin({
      assignments: this.http.get<TeamAssignmentSubmission[]>(this.apiService.managerTeamAssignmentsSubmissionsUrl, { headers }).pipe(
        catchError(() => of([]))
      ),
      feedback: this.http.get<TeamFeedbackSubmission[]>(this.apiService.managerTeamFeedbackSubmissionsUrl, { headers }).pipe(
        catchError(() => of([]))
      )
    }).subscribe({
      next: (results) => {
        this.teamAssignmentSubmissions = results.assignments;
        this.teamFeedbackSubmissions = results.feedback;
        this.isLoadingSubmissions = false;
      },
      error: (err) => {
        console.error('Failed to fetch team submissions:', err);
        this.teamAssignmentSubmissions = [];
        this.teamFeedbackSubmissions = [];
        this.isLoadingSubmissions = false;
      }
    });
  }

  // Open feedback modal for a submission
  openFeedbackModal(submission: TeamAssignmentSubmission | TeamFeedbackSubmission, type: 'assignment' | 'feedback'): void {
    this.selectedSubmissionForFeedback = { type, submission };
    this.performanceFeedback = {
      training_id: submission.training_id,
      employee_empid: submission.employee_empid,
      application_of_training: null,
      quality_of_deliverables: null,
      problem_solving_capability: null,
      productivity_independence: null,
      process_compliance_adherence: null,
      improvement_areas: '',
      strengths: '',
      overall_performance: 0,
      additional_comments: ''
    };
    this.existingFeedback = null;
    this.showFeedbackModal = true;
    
    // Load existing feedback if any
    this.loadExistingFeedback(submission.training_id, submission.employee_empid);
  }

  // Load existing feedback (latest) and all feedback history
  loadExistingFeedback(trainingId: number, employeeEmpid: string): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    // Load latest feedback for editing
    this.http.get<ManagerPerformanceFeedback>(this.apiService.managerPerformanceFeedbackByIdUrl(trainingId, employeeEmpid), { headers }).subscribe({
      next: (feedback) => {
        if (feedback) {
          this.existingFeedback = feedback;
          this.performanceFeedback = {
            training_id: feedback.training_id,
            employee_empid: feedback.employee_empid,
            application_of_training: feedback.application_of_training || null,
            quality_of_deliverables: feedback.quality_of_deliverables || null,
            problem_solving_capability: feedback.problem_solving_capability || null,
            productivity_independence: feedback.productivity_independence || null,
            process_compliance_adherence: feedback.process_compliance_adherence || null,
            improvement_areas: feedback.improvement_areas || '',
            strengths: feedback.strengths || '',
            overall_performance: 0,
            additional_comments: feedback.additional_comments || ''
          };
          // Calculate overall performance from the loaded ratings
          this.calculateOverallPerformance();
        }
      },
      error: (err) => {
        // No existing feedback, that's okay
        console.log('No existing feedback found');
      }
    });

    // Load all feedback history
    this.isLoadingFeedbackHistory = true;
    this.feedbackHistory = [];
    this.http.get<ManagerPerformanceFeedback[]>(this.apiService.managerPerformanceFeedbackHistoryUrl(trainingId, employeeEmpid), { headers }).subscribe({
      next: (history) => {
        // Get training information to enrich feedback with skill_category
        const training = this.allTrainings.find(t => t.id === trainingId);
        const skillCategory = training?.skill_category || 'Unknown';

        // Enrich feedback with skill_category and sort by updated_at descending (most recent first)
        const enrichedHistory = (history || []).map(feedback => ({
          ...feedback,
          skill_category: skillCategory
        })).sort((a, b) => {
          const dateA = new Date(a.updated_at || a.created_at || 0).getTime();
          const dateB = new Date(b.updated_at || b.created_at || 0).getTime();
          return dateB - dateA;
        });

        // Add update numbers to each entry
        this.feedbackHistory = enrichedHistory.map((feedback, index) => ({
          ...feedback,
          updateNumber: index + 1,
          totalUpdates: enrichedHistory.length
        }));

        // Group feedback by skill type
        this.feedbackHistoryByType = this.groupFeedbackBySkillType(this.feedbackHistory);
        
        this.isLoadingFeedbackHistory = false;
      },
      error: (err) => {
        console.error('Failed to load feedback history:', err);
        this.feedbackHistory = [];
        this.feedbackHistoryByType = new Map();
        this.isLoadingFeedbackHistory = false;
      }
    });
  }

  // Group feedback by skill type (L1-L5)
  groupFeedbackBySkillType(feedbackList: ManagerPerformanceFeedback[]): Map<string, ManagerPerformanceFeedback[]> {
    const grouped = new Map<string, ManagerPerformanceFeedback[]>();
    
    feedbackList.forEach(feedback => {
      // Get skill category (L1, L2, L3, L4, L5) or default to 'Unknown'
      const skillType = feedback.skill_category || 'Unknown';
      
      if (!grouped.has(skillType)) {
        grouped.set(skillType, []);
      }
      grouped.get(skillType)!.push(feedback);
    });
    
    // Sort feedback within each group by updated_at (most recent first)
    grouped.forEach((feedbackList, skillType) => {
      feedbackList.sort((a, b) => {
        const dateA = new Date(a.updated_at || a.created_at || 0).getTime();
        const dateB = new Date(b.updated_at || b.created_at || 0).getTime();
        return dateB - dateA;
      });
    });
    
    return grouped;
  }

  // Get latest feedback for a skill type
  getLatestFeedbackForType(feedbackList: ManagerPerformanceFeedback[]): ManagerPerformanceFeedback | null {
    if (!feedbackList || feedbackList.length === 0) return null;
    
    // List is already sorted by updated_at descending
    return feedbackList[0];
  }

  // Get skill type display name
  getSkillTypeDisplayName(skillType: string): string {
    const typeMap: { [key: string]: string } = {
      'L1': 'Beginner',
      'L2': 'Basic',
      'L3': 'Intermediate',
      'L4': 'Advanced',
      'L5': 'Expert'
    };
    return typeMap[skillType] || skillType;
  }

  // Get skill type entries as array for template iteration
  getSkillTypeEntries(): Array<{key: string, value: ManagerPerformanceFeedback[]}> {
    return Array.from(this.feedbackHistoryByType.entries()).map(([key, value]) => ({
      key,
      value
    }));
  }

  // Open feedback history modal for a specific skill type
  openFeedbackHistoryModal(skillType: string): void {
    if (!skillType || !this.selectedSubmissionForFeedback) return;
    this.selectedSkillTypeForHistory = skillType;
    this.selectedTrainingNameForHistory = this.selectedSubmissionForFeedback.submission.training_name;
    this.selectedEmployeeNameForHistory = this.selectedSubmissionForFeedback.submission.employee_name;
    
    // Get all feedback for this skill type
    this.feedbackHistoryList = this.feedbackHistory.filter(fb => 
      (fb.skill_category || 'Unknown') === skillType
    );
    
    // Sort by updated_at descending (most recent first)
    this.feedbackHistoryList.sort((a, b) => {
      const dateA = new Date(a.updated_at || a.created_at || 0).getTime();
      const dateB = new Date(b.updated_at || b.created_at || 0).getTime();
      return dateB - dateA;
    });
    
    this.showFeedbackHistoryModal = true;
  }

  // Close feedback history modal
  closeFeedbackHistoryModal(): void {
    this.showFeedbackHistoryModal = false;
    this.selectedSkillTypeForHistory = '';
    this.selectedTrainingNameForHistory = '';
    this.selectedEmployeeNameForHistory = '';
    this.feedbackHistoryList = [];
  }

  // Close feedback modal
  closeFeedbackModal(): void {
    this.showFeedbackModal = false;
    this.selectedSubmissionForFeedback = null;
    this.existingFeedback = null;
    this.feedbackHistory = [];
    this.feedbackHistoryByType = new Map();
    this.performanceFeedback = {
      training_id: 0,
      employee_empid: '',
      application_of_training: null,
      quality_of_deliverables: null,
      problem_solving_capability: null,
      productivity_independence: null,
      process_compliance_adherence: null,
      improvement_areas: '',
      strengths: '',
      overall_performance: 0,
      additional_comments: ''
    };
  }

  // Open final evaluation modal from submission card
  openFinalEvaluationFromCard(submission: TeamAssignmentSubmission | TeamFeedbackSubmission): void {
    // Determine submission type by checking if it has 'score' property (assignment) or 'responses' property (feedback)
    const submissionType: 'assignment' | 'feedback' = 'score' in submission ? 'assignment' : 'feedback';
    
    // Set the selected submission for feedback
    this.selectedSubmissionForFeedback = { 
      type: submissionType, 
      submission: submission 
    };
    // Open the final evaluation modal
    this.openFinalEvaluationModal();
  }

  // Open final evaluation modal
  openFinalEvaluationModal(): void {
    if (!this.selectedSubmissionForFeedback) return;
    
    this.showFinalEvaluationModal = true;
    this.isLoadingEmployeeSkills = true;
    this.employeeSkillsForEvaluation = [];
    this.isManagerSatisfied = null; // Reset satisfaction status
    
    const employeeEmpid = this.selectedSubmissionForFeedback.submission.employee_empid;
    
    // Try to get skills from already loaded manager data
    if (this.manager) {
      const teamMember = this.manager.team.find(member => member.id === employeeEmpid);
      if (teamMember && teamMember.skills) {
        this.employeeSkillsForEvaluation = teamMember.skills.map((skill: Competency) => ({
          skill: skill.skill,
          current_expertise: skill.current_expertise,
          target_expertise: skill.target_expertise,
          targetAchieved: false // Initially unchecked
        }));
        this.isLoadingEmployeeSkills = false;
        return;
      }
    }
    
    // If not found in loaded data, fetch manager dashboard data
    const token = this.authService.getToken();
    if (!token) {
      this.isLoadingEmployeeSkills = false;
      return;
    }
    
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    // Fetch manager dashboard data to get team member skills
    this.http.get<any>(this.apiService.getUrl('/data/manager/dashboard'), { headers }).subscribe({
      next: (data) => {
        if (data && data.team) {
          const teamMember = data.team.find((member: TeamMember) => member.id === employeeEmpid);
          if (teamMember && teamMember.skills) {
            this.employeeSkillsForEvaluation = teamMember.skills.map((skill: Competency) => ({
              skill: skill.skill,
              current_expertise: skill.current_expertise,
              target_expertise: skill.target_expertise,
              targetAchieved: false // Initially unchecked
            }));
          }
        }
        this.isLoadingEmployeeSkills = false;
      },
      error: (err) => {
        console.error('Failed to fetch employee skills:', err);
        this.toastService.error('Failed to load employee skills for evaluation');
        this.isLoadingEmployeeSkills = false;
      }
    });
  }

  // Close final evaluation modal
  closeFinalEvaluationModal(): void {
    this.showFinalEvaluationModal = false;
    this.employeeSkillsForEvaluation = [];
    this.isManagerSatisfied = null; // Reset satisfaction status
  }

  // Get count of achieved skills
  getAchievedSkillsCount(): number {
    return this.employeeSkillsForEvaluation.filter(skill => skill.targetAchieved).length;
  }

  // Submit final evaluation
  submitFinalEvaluation(): void {
    if (!this.selectedSubmissionForFeedback) return;
    
    // Check if manager is satisfied
    if (this.isManagerSatisfied === null) {
      this.toastService.warning('Please indicate whether you are satisfied with the candidate\'s performance');
      return;
    }
    
    if (this.isManagerSatisfied === false) {
      // Should not reach here if UI is correct, but handle it anyway
      this.reassignTraining();
      return;
    }
    
    const achievedSkills = this.employeeSkillsForEvaluation.filter(skill => skill.targetAchieved);
    if (achievedSkills.length === 0) {
      this.toastService.error('Please select at least one skill that has achieved its target state');
      return;
    }

    this.isSubmittingEvaluation = true;
    
    const token = this.authService.getToken();
    if (!token) {
      this.isSubmittingEvaluation = false;
      return;
    }
    
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' });
    const employeeEmpid = this.selectedSubmissionForFeedback.submission.employee_empid;
    
    // Update each achieved skill
    const updatePromises = achievedSkills.map(skill => {
      const payload = {
        employee_username: employeeEmpid,
        skill_name: skill.skill,
        current_expertise: skill.target_expertise, // Update current to match target
        target_expertise: skill.target_expertise
      };
      
      return this.http.put(this.apiService.getUrl('/data/manager/team-skill'), payload, { headers }).toPromise();
    });

    // Execute all updates
    Promise.all(updatePromises).then(() => {
      this.isSubmittingEvaluation = false;
      
      // Show success message
      this.successPopupMessage = `Successfully updated ${achievedSkills.length} skill${achievedSkills.length !== 1 ? 's' : ''} and awarded ${achievedSkills.length} badge${achievedSkills.length !== 1 ? 's' : ''}!`;
      this.showSuccessPopup = true;
      
      // Auto-close popup after 4 seconds
      setTimeout(() => {
        if (this.showSuccessPopup) {
          this.closeSuccessPopup();
        }
      }, 4000);
      
      // Close evaluation modal
      this.closeFinalEvaluationModal();
      
      // Refresh team data to reflect changes
      this.fetchDashboardData();
      
      this.toastService.success(`Final evaluation completed! ${achievedSkills.length} badge${achievedSkills.length !== 1 ? 's' : ''} awarded.`);
    }).catch((err) => {
      this.isSubmittingEvaluation = false;
      console.error('Failed to submit final evaluation:', err);
      if (err.status === 403) {
        this.toastService.error('You can only update skills for employees in your team.');
      } else {
        this.toastService.error(`Failed to submit final evaluation. Error: ${err.statusText || 'Unknown error'}`);
      }
    });
  }

  // Reassign training when manager is not satisfied
  reassignTraining(): void {
    if (!this.selectedSubmissionForFeedback) return;
    
    const trainingId = this.selectedSubmissionForFeedback.submission.training_id;
    const employeeEmpid = this.selectedSubmissionForFeedback.submission.employee_empid;
    const trainingName = this.selectedSubmissionForFeedback.submission.training_name;
    const employeeName = this.selectedSubmissionForFeedback.submission.employee_name;
    
    if (!trainingId || !employeeEmpid) {
      this.toastService.error('Missing training or employee information');
      return;
    }
    
    this.isReassigningTraining = true;
    
    const token = this.authService.getToken();
    if (!token) {
      this.isReassigningTraining = false;
      this.toastService.error('Authentication token missing. Please login again.');
      return;
    }
    
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    // First, delete the existing assignment
    this.http.delete(this.apiService.getUrl(`/assignments/${trainingId}/${employeeEmpid}`), { headers }).subscribe({
      next: () => {
        // After deletion, recreate the assignment
        const payload = { 
          training_id: trainingId, 
          employee_username: employeeEmpid 
        };
        
        this.http.post(this.apiService.assignmentsUrl, payload, { headers }).subscribe({
          next: () => {
            this.isReassigningTraining = false;
            
            // Show success message
            this.successPopupMessage = `Training "${trainingName}" has been reassigned to ${employeeName}. They can now retake the training.`;
            this.showSuccessPopup = true;
            
            // Auto-close popup after 4 seconds
            setTimeout(() => {
              if (this.showSuccessPopup) {
                this.closeSuccessPopup();
              }
            }, 4000);
            
            // Close evaluation modal
            this.closeFinalEvaluationModal();
            
            // Refresh team data to reflect changes
            this.fetchDashboardData();
            this.fetchTeamAssignedTrainings();
            this.fetchTeamSubmissions();
            
            this.toastService.success(`Training reassigned successfully to ${employeeName}`);
          },
          error: (err) => {
            this.isReassigningTraining = false;
            console.error('Failed to recreate assignment:', err);
            this.toastService.error(`Failed to reassign training. Error: ${err.statusText || 'Unknown error'}`);
          }
        });
      },
      error: (err) => {
        this.isReassigningTraining = false;
        console.error('Failed to delete assignment:', err);
        if (err.status === 404) {
          // Assignment might not exist, try to create it anyway
          const payload = { 
            training_id: trainingId, 
            employee_username: employeeEmpid 
          };
          
          this.http.post(this.apiService.assignmentsUrl, payload, { headers }).subscribe({
            next: () => {
              this.isReassigningTraining = false;
              this.successPopupMessage = `Training "${trainingName}" has been reassigned to ${employeeName}.`;
              this.showSuccessPopup = true;
              setTimeout(() => {
                if (this.showSuccessPopup) {
                  this.closeSuccessPopup();
                }
              }, 4000);
              this.closeFinalEvaluationModal();
              this.fetchDashboardData();
              this.fetchTeamAssignedTrainings();
              this.fetchTeamSubmissions();
              this.toastService.success(`Training reassigned successfully to ${employeeName}`);
            },
            error: (createErr) => {
              this.isReassigningTraining = false;
              this.toastService.error(`Failed to reassign training. Error: ${createErr.statusText || 'Unknown error'}`);
            }
          });
        } else {
          this.toastService.error(`Failed to delete assignment. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  // Update submission feedback status locally
  updateSubmissionFeedbackStatus(trainingId: number, employeeEmpid: string, hasFeedback: boolean): void {
    // Update assignment submissions
    const assignmentSubmission = this.teamAssignmentSubmissions.find(
      s => s.training_id === trainingId && s.employee_empid === employeeEmpid
    );
    if (assignmentSubmission) {
      assignmentSubmission.has_feedback = hasFeedback;
      assignmentSubmission.feedback_count = hasFeedback ? 1 : 0;
    }

    // Update feedback submissions
    const feedbackSubmission = this.teamFeedbackSubmissions.find(
      s => s.training_id === trainingId && s.employee_empid === employeeEmpid
    );
    if (feedbackSubmission) {
      feedbackSubmission.has_feedback = hasFeedback;
      feedbackSubmission.feedback_count = hasFeedback ? 1 : 0;
    }
  }

  // Close success popup
  closeSuccessPopup(): void {
    this.showSuccessPopup = false;
    this.successPopupMessage = '';
  }
  
  closeAttendanceSuccessPopup(): void {
    this.showAttendanceSuccessPopup = false;
    this.attendanceSuccessData = null;
  }

  // Calculate overall performance as average of the five ratings
  // Only calculates when all five ratings are provided
  calculateOverallPerformance(): void {
    // Get all five ratings
    const r1 = this.performanceFeedback.application_of_training;
    const r2 = this.performanceFeedback.quality_of_deliverables;
    const r3 = this.performanceFeedback.problem_solving_capability;
    const r4 = this.performanceFeedback.productivity_independence;
    const r5 = this.performanceFeedback.process_compliance_adherence;

    // Check if all five ratings are provided (not null or undefined)
    if (r1 === null || r1 === undefined || r2 === null || r2 === undefined || 
        r3 === null || r3 === undefined || r4 === null || r4 === undefined || 
        r5 === null || r5 === undefined) {
      this.performanceFeedback.overall_performance = 0;
      return;
    }

    // Convert to numbers explicitly (handle both number and string inputs)
    const rating1 = typeof r1 === 'string' ? parseInt(r1, 10) : Number(r1);
    const rating2 = typeof r2 === 'string' ? parseInt(r2, 10) : Number(r2);
    const rating3 = typeof r3 === 'string' ? parseInt(r3, 10) : Number(r3);
    const rating4 = typeof r4 === 'string' ? parseInt(r4, 10) : Number(r4);
    const rating5 = typeof r5 === 'string' ? parseInt(r5, 10) : Number(r5);

    // Verify all are valid numbers
    if (isNaN(rating1) || isNaN(rating2) || isNaN(rating3) || isNaN(rating4) || isNaN(rating5)) {
      this.performanceFeedback.overall_performance = 0;
      return;
    }

    // Calculate sum of all five ratings
    const sum = rating1 + rating2 + rating3 + rating4 + rating5;
    
    // Calculate average: sum divided by 5
    const average = sum / 5;
    
    // Round to nearest integer (1-5 scale)
    this.performanceFeedback.overall_performance = Math.round(average);
    
    // Ensure it's within valid range (should always be 1-5 after rounding)
    if (this.performanceFeedback.overall_performance < 1) {
      this.performanceFeedback.overall_performance = 1;
    } else if (this.performanceFeedback.overall_performance > 5) {
      this.performanceFeedback.overall_performance = 5;
    }
  }

  // Get overall performance display text
  getOverallPerformanceText(): string {
    if (this.performanceFeedback.overall_performance === 0) {
      return 'Not Calculated (Please provide all ratings)';
    }
    const ratingLabels: { [key: number]: string } = {
      1: '1 - Poor',
      2: '2 - Below Average',
      3: '3 - Average',
      4: '4 - Good',
      5: '5 - Excellent'
    };
    return ratingLabels[this.performanceFeedback.overall_performance] || 'Not Calculated';
  }

  // Submit performance feedback
  submitPerformanceFeedback(): void {
    if (!this.selectedSubmissionForFeedback) return;
    
    // Ensure overall performance is calculated
    this.calculateOverallPerformance();
    
    // Validate that all five ratings are provided
    const ratings = [
      { name: 'Application of Training in Daily Work', value: this.performanceFeedback.application_of_training },
      { name: 'Quality of Deliverables', value: this.performanceFeedback.quality_of_deliverables },
      { name: 'Problem-Solving Capability', value: this.performanceFeedback.problem_solving_capability },
      { name: 'Productivity & Independence', value: this.performanceFeedback.productivity_independence },
      { name: 'Process & Compliance Adherence', value: this.performanceFeedback.process_compliance_adherence }
    ];

    // Check if all ratings are provided
    const missingRatings = ratings.filter(rating => rating.value === null || rating.value === undefined);
    if (missingRatings.length > 0) {
      this.toastService.error('Please provide all five performance ratings before submitting.');
      return;
    }

    // Validate rating values are within range
    for (const rating of ratings) {
      if (rating.value !== null && (rating.value < 1 || rating.value > 5)) {
        this.toastService.error(`${rating.name} rating must be between 1 and 5`);
        return;
      }
    }

    // Validate overall performance is calculated correctly
    if (this.performanceFeedback.overall_performance < 1 || this.performanceFeedback.overall_performance > 5) {
      this.toastService.error('Overall performance rating must be between 1 and 5');
      return;
    }


    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' });
    
    this.isSubmittingFeedback = true;
    
    const payload = {
      training_id: this.performanceFeedback.training_id,
      employee_empid: this.performanceFeedback.employee_empid,
      application_of_training: this.performanceFeedback.application_of_training,
      quality_of_deliverables: this.performanceFeedback.quality_of_deliverables,
      problem_solving_capability: this.performanceFeedback.problem_solving_capability,
      productivity_independence: this.performanceFeedback.productivity_independence,
      process_compliance_adherence: this.performanceFeedback.process_compliance_adherence,
      improvement_areas: this.performanceFeedback.improvement_areas || null,
      strengths: this.performanceFeedback.strengths || null,
      overall_performance: this.performanceFeedback.overall_performance,
      additional_comments: this.performanceFeedback.additional_comments || null
    };

    this.http.post<ManagerPerformanceFeedback>(this.apiService.managerPerformanceFeedbackUrl, payload, { headers }).subscribe({
      next: (response) => {
        this.isSubmittingFeedback = false;
        const isUpdate = this.existingFeedback !== null;
        
        // Update local status immediately
        this.updateSubmissionFeedbackStatus(
          this.performanceFeedback.training_id,
          this.performanceFeedback.employee_empid,
          true
        );
        
        // Store training and employee info before closing modal
        const trainingId = this.performanceFeedback.training_id;
        const employeeEmpid = this.performanceFeedback.employee_empid;
        
        // Close feedback modal first
        this.closeFeedbackModal();
        
        // Show success popup after a brief delay to ensure modal is closed
        setTimeout(() => {
          this.successPopupMessage = 'Feedback submitted successfully! A new entry has been created. All previous feedback entries are preserved and visible.';
          this.showSuccessPopup = true;
          console.log('Success popup should be visible now:', this.showSuccessPopup, this.successPopupMessage);
          
          // Auto-close popup after 4 seconds
          setTimeout(() => {
            if (this.showSuccessPopup) {
              this.closeSuccessPopup();
            }
          }, 4000);
        }, 300);
        
        // Refresh team submissions to update status from backend
        this.fetchTeamSubmissions();
      },
      error: (err) => {
        this.isSubmittingFeedback = false;
        console.error('Failed to submit feedback:', err);
        if (err.status === 403) {
          this.toastService.error('You can only provide feedback for employees in your team.');
        } else if (err.status === 400) {
          this.toastService.error(err.error?.detail || 'Invalid feedback data. Please check your ratings.');
        } else {
          this.toastService.error(`Failed to submit feedback. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }
  
  openScheduleTrainingModal(): void {
    this.newTraining.trainer_name = this.managerDisplayName || this.manager?.name || this.manager?.id || this.authService.getUsername() || '';
    this.showScheduleTrainingModal = true;
  }

  closeScheduleTrainingModal(): void {
    this.showScheduleTrainingModal = false;
    this.newTraining = {
        division: '',
        department: '',
        competency: '',
        skill: '',
        training_name: '',
        training_topics: '',
        prerequisites: '',
        skill_category: 'L1',
        trainer_name: '',
        email: '',
        training_date: '',
        duration: '',
        time: '',
        training_type: 'Online',
        seats: '',
        assessment_details: ''
    };
  }

  // New Modal Methods
  openDetailModal(type: 'mySkillsMet' | 'mySkillGaps' | 'teamSkillsMet' | 'teamSkillGaps' | 'totalMembers' | 'additionalSkills' | 'coreSkills') {
    if (!this.manager) return;
    
    this.modalDataType = null;

    switch(type) {
      case 'totalMembers':
        this.modalTitle = 'Total Team Members';
        this.modalData = this.manager.team;
        this.modalDataType = 'members';
        break;
      case 'coreSkills':
        this.modalTitle = 'Core Skills';
        // For Core Skills, we show all core skills without status
        this.modalData = this.sections.map((section, index) => ({
          id: index + 1,
          skill: section.title,
          competency: section.subtitle || 'Core Competency'
        }));
        this.modalDataType = 'skills';
        break;
      case 'mySkillsMet':
        this.modalTitle = 'My Skills Met';
        this.modalData = this.manager.skills.filter(s => s.status === 'Met');
        this.modalDataType = 'skills';
        break;
      case 'mySkillGaps':
        this.modalTitle = 'My Skill Gaps';
        this.modalData = this.manager.skills.filter(s => s.status === 'Gap');
        this.modalDataType = 'skills';
        break;
      case 'teamSkillsMet':
        this.modalTitle = 'Team Skills Met';
        this.modalData = this.manager.team.flatMap(m => m.skills.filter(s => s.status === 'Met').map(s => ({...s, memberName: m.name})));
        this.modalDataType = 'skills';
        break;
      case 'teamSkillGaps':
        this.modalTitle = 'Team Skill Gaps';
        this.modalData = this.manager.team.flatMap(m => m.skills.filter(s => s.status === 'Gap').map(s => ({...s, memberName: m.name})));
        this.modalDataType = 'skills';
        break;
      case 'additionalSkills':
          this.modalTitle = 'My Additional Skills';
          this.modalData = this.additionalSkills;
          this.modalDataType = 'skills'; // Reuse skills template for this
          break;
    }
    this.showDetailModal = true;
  }

  closeDetailModal() {
    this.showDetailModal = false;
    this.modalTitle = '';
    this.modalData = [];
    this.modalDataType = null;
  }


  scheduleTraining(): void {
    const token = this.authService.getToken();
    if (!token) {
        this.errorMessage = 'Authentication error. Please log in again.';
        return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });

    const payload = {
      division: this.newTraining.division || null,
      department: this.newTraining.department || null,
      training_name: this.newTraining.training_name,
      training_topics: this.newTraining.training_topics,
      prerequisites: this.newTraining.prerequisites,
      skill_category: this.newTraining.skill_category,
      trainer_name: this.newTraining.trainer_name,
      email: this.newTraining.email,
      training_date: this.newTraining.training_date || null,
      duration: this.newTraining.duration,
      time: this.newTraining.time,
      training_type: this.newTraining.training_type,
      seats: this.newTraining.seats,
      assessment_details: this.newTraining.assessment_details
    };

    this.http.post(this.apiService.trainingsUrl, payload, { headers }).subscribe({
        next: (response) => {
            this.toastService.success('Training scheduled successfully!');
            this.closeScheduleTrainingModal();
            this.fetchTrainingCatalog();
            this.fetchScheduledTrainings();
        },
        error: (err) => {
            if (err.status === 401) {
              this.toastService.error('Your session has expired. Please log in again.');
              this.authService.logout();
              this.router.navigate(['/login']);
            } else if (err.status === 422 && err.error && err.error.detail) {
              const errorDetails = err.error.detail.map((e: any) => `- Field '${e.loc[1]}': ${e.msg}`).join('\n');
              const fullErrorMessage = `Please correct the following errors:\n${errorDetails}`;
              this.errorMessage = fullErrorMessage;
              this.toastService.error(fullErrorMessage);
            } else {
              const detail = err.error?.detail || 'An unknown error occurred. Please try again.';
              this.errorMessage = `Failed to schedule training: ${detail}`;
              this.toastService.error(this.errorMessage);
            }
        }
    });
  }

  approveRequest(request: TrainingRequest, notes?: string): void {
    this.respondToRequest(request, 'approved', notes);
  }

  rejectRequest(request: TrainingRequest, notes?: string): void {
    this.respondToRequest(request, 'rejected', notes);
  }

  respondToRequestById(requestId: number, status: 'approved' | 'rejected'): void {
    const request = this.pendingRequests.find(r => r.id === requestId);
    if (request) {
      this.respondToRequest(request, status);
    }
  }

  private respondToRequest(request: TrainingRequest, status: 'approved' | 'rejected', notes?: string): void {
    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const responseData = {
      status: status,
      manager_notes: notes || null
    };

    this.http.put(this.apiService.trainingRequestRespondUrl(request.id), responseData, { headers }).subscribe({
      next: (response) => {
        this.toastService.success(`Training request ${status} successfully!`);
        this.fetchPendingRequests(); // Refresh the requests list
      },
      error: (err) => {
        console.error(`Failed to ${status} training request:`, err);
        this.toastService.error(`Failed to ${status} training request. Error: ${err.statusText || 'Unknown error'}`);
      }
    });
  }

  fetchTrainingCatalog(): void {
    const token = this.authService.getToken();
    if (!token) {
        return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });

    this.http.get<TrainingDetail[]>(this.apiService.trainingsUrl, { headers }).subscribe({
      next: (data) => {
        console.log('Training catalog data loaded:', data);
        this.trainingCatalog = data;
        this.allTrainings = data; // Align with engineer dashboard
        this.allTrainingsCalendarEvents = this.allTrainings
          .filter(t => t.training_date)
          .map(t => ({
            date: new Date(t.training_date as string),
            title: t.training_name,
            trainer: t.trainer_name || 'N/A'
          }));
      },
      error: (err) => {
        console.error('Failed to load training catalog:', err);
        this.errorMessage = 'Failed to load training catalog.';
      }
    });
  }

  fetchScheduledTrainings(): void {
    const token = this.authService.getToken();
    if (!token) {
        return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });

    this.http.get<TrainingDetail[]>(this.apiService.trainingsUrl, { headers }).subscribe({
      next: (response) => {
        this.allTrainings = response;
        // Clear cache when trainings are updated
        this._myTrainingsCache = [];
        this._myTrainingsCacheKey = '';
        this.allTrainingsCalendarEvents = this.allTrainings
            .filter(t => t.training_date)
            .map(t => ({
                date: new Date(t.training_date as string),
                title: t.training_name,
                trainer: t.trainer_name || 'N/A'
            }));
      },
      error: (err) => {
        console.error('Failed to fetch scheduled trainings:', err);
      }
    });
  }

  fetchPendingRequests(): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.http.get<TrainingRequest[]>(this.apiService.pendingTrainingRequestsUrl, { headers }).subscribe({
      next: (response) => {
        this.pendingRequests = response || [];
      },
      error: (err) => {
        console.error('Failed to fetch pending requests:', err);
      }
    });
  }

  fetchAssignedTrainings(): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    // For managers, only fetch personal assigned trainings (assigned to the manager by their manager)
    // Team assigned trainings should not be shown in the "Assigned Trainings" tab
    this.http.get<TrainingDetail[]>(this.apiService.myAssignmentsUrl, { headers }).subscribe({
      next: (response) => {
        console.log('Personal assigned trainings loaded:', response);
        this.assignedTrainings = (response || []).map(t => ({ ...t, assignmentType: 'personal' as const }));
        this.assignedTrainingsCalendarEvents = this.assignedTrainings
          .filter(t => t.training_date)
          .map(t => ({
            date: new Date(t.training_date as string),
            title: t.training_name,
            trainer: t.trainer_name || 'N/A'
          }));
        this.generateCalendar();
      },
      error: (err) => {
        console.error('Failed to fetch assigned trainings:', err);
        this.assignedTrainings = [];
        this.assignedTrainingsCalendarEvents = [];
      }
    });
  }

  get filteredCatalog(): TrainingDetail[] {
    let list = [...(this.trainingCatalog || [])];
    if (this.catalogSearch && this.catalogSearch.trim()) {
      const q = this.catalogSearch.trim().toLowerCase();
      list = list.filter(t =>
        (t.training_name || '').toLowerCase().includes(q) ||
        (t.training_topics || '').toLowerCase().includes(q) ||
        (t.trainer_name || '').toLowerCase().includes(q) ||
        (t.skill || '').toLowerCase().includes(q)
      );
    }
    if (this.catalogTypeFilter !== 'All') {
      list = list.filter(t => (t.training_type || '').toLowerCase() === this.catalogTypeFilter.toLowerCase());
    }
    if (this.catalogCategoryFilter !== 'All') {
      list = list.filter(t => (t.skill_category || '').toLowerCase() === this.catalogCategoryFilter.toLowerCase());
    }
    
    list.sort((a, b) => {
      const dateA = a.training_date ? new Date(a.training_date).getTime() : Infinity;
      const dateB = b.training_date ? new Date(b.training_date).getTime() : Infinity;
      return dateA - dateB;
    });
    
    return list;
  }

  get myTrainings(): TrainingDetail[] {
    // Backend stores trainer_name as username (manager id), so prioritize matching against manager id
    const managerId = this.manager?.id || this.authService.getUsername() || '';
    const managerName = this.managerDisplayName || this.manager?.name || '';
    
    if (!managerId && !managerName) {
      return [];
    }
    
    // Use cache to prevent repeated calculations during change detection
    const cacheKey = `${managerId}-${managerName}-${this.allTrainings.length}`;
    if (this._myTrainingsCacheKey === cacheKey && this._myTrainingsCache.length >= 0) {
      return this._myTrainingsCache;
    }
    
    const mgrId = String(managerId).trim();
    const mgrName = (managerName || '').trim();
    
    // Filter: Match against managerId first (what backend stores), then managerName as fallback
    const filtered = this.allTrainings
      .filter(t => {
        const trainerName = String(t.trainer_name || '').trim();
        if (!trainerName) return false;
        
        // Normalize all strings for comparison
        const trainerNameLower = trainerName.toLowerCase();
        const mgrIdLower = mgrId.toLowerCase();
        const mgrNameLower = mgrName.toLowerCase();
        
        // Primary match: managerId (username) - exact match (case-insensitive)
        const matchesId = trainerNameLower === mgrIdLower;
        
        // Fallback match: managerName (for trainings imported via Excel or other sources)
        const matchesName = mgrName && mgrNameLower.length > 0 && trainerNameLower === mgrNameLower;
        
        // Additional: Check if trainer_name contains the managerId (for partial matches)
        const containsId = mgrIdLower.length > 0 && trainerNameLower.includes(mgrIdLower);
        
        // Additional: Check if trainer_name contains the managerName (for partial matches)
        const containsName = mgrName && mgrNameLower.length > 0 && trainerNameLower.includes(mgrNameLower);
        
        return matchesId || matchesName || containsId || containsName;
      })
      .sort((a, b) => {
        const dateA = a.training_date ? new Date(a.training_date).getTime() : 0;
        const dateB = b.training_date ? new Date(b.training_date).getTime() : 0;
        return dateB - dateA; // Sort descending
      });
    
    // Update cache
    this._myTrainingsCache = filtered;
    this._myTrainingsCacheKey = cacheKey;
    // Check shared status for all trainings and fetch candidates
    filtered.forEach(training => {
      this.checkSharedStatus(training.id);
      this.fetchTrainingCandidates(training.id);
    });
    
    return filtered;
  }

  isUpcoming(dateStr?: string): boolean {
    if (!dateStr) return false;
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return new Date(dateStr) >= today;
  }

  fetchDashboardData(): void {
    this.isLoading = true;
    const token = this.authService.getToken();
    const userRole = this.authService.getRole();
      
    if (!token || userRole !== 'manager') {
      this.errorMessage = 'Authentication token not found or invalid role. Please log in again.';
      this.isLoading = false;
      this.router.navigate(['/login']);
      return;
    }
      
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });

    this.http.get<any>(this.apiService.getUrl(this.API_ENDPOINT), { headers }).subscribe({
      next: (response) => {
        this.manager = response;
        const anyResp: any = response as any;
        this.managerDisplayName = (anyResp.employee_name && String(anyResp.employee_name).trim())
          || (response.name && String(response.name).trim())
          || (response.id && String(response.id).trim())
          || '';
        this.managerIsTrainer = !!response.manager_is_trainer;
        this.processDashboardData();
        this.extractUniqueMySkills();
        this.extractUniqueTeamSkills();
        this.extractUniqueTeamMembers();
        this.extractUniqueCompetencies();
        this.extractUniqueCurrentLevels();
        this.loadAdditionalSkills();
        this.isLoading = false;
        this.errorMessage = '';
      },
      error: (err) => {
        this.errorMessage = 'Failed to load dashboard data. Please ensure the API is running and the endpoint is correct.';
        console.error('Error fetching data:', err);
        this.isLoading = false;
      }
    });
  }

  processDashboardData(): void {
    if (!this.manager) return;
    this.totalTeamMembers = this.manager.team.length;
    const allTeamSkills = this.manager.team.flatMap((member: TeamMember) => member.skills);
    this.teamSkillsMet = allTeamSkills.filter((skill: Competency) => skill.status === 'Met').length;
    this.teamSkillsGap = allTeamSkills.filter((skill: Competency) => skill.status === 'Gap').length;
    
    const skillGapCount: { [key: string]: number } = {};
    allTeamSkills.forEach((skill: Competency) => {
      if (skill.status === 'Gap') {
        skillGapCount[skill.skill] = (skillGapCount[skill.skill] || 0) + 1;
      }
    });

    this.topSkillGaps = Object.entries(skillGapCount)
      .map(([skill, count]) => ({ skill, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 3); // Limit to top 3 for a more compact view
  }
  
  extractUniqueMySkills(): void {
    if (this.manager && this.manager.skills) {
      this.uniqueMySkills = [...new Set(this.manager.skills.map(skill => skill.skill))];
      this.skillNames = Array.from(new Set(this.manager.skills.map(skill => skill.skill))).sort();
    }
  }

  extractUniqueTeamSkills(): void {
    if (this.manager && this.manager.team) {
      const allSkills = this.manager.team.flatMap(member => member.skills.map(skill => skill.skill));
      this.uniqueTeamSkills = [...new Set(allSkills)];
    }
  }

  extractUniqueTeamMembers(): void {
    if (this.manager && this.manager.team) {
      this.uniqueTeamMembers = [...new Set(this.manager.team.map(member => member.name))];
    }
  }

  extractUniqueCompetencies(): void {
    if (this.manager && this.manager.team) {
      const allCompetencies = this.manager.team.flatMap(member =>
        member.skills.map(skill => skill.competency).filter(comp => comp)
      );
      this.uniqueCompetencies = [...new Set(allCompetencies)];
    }
  }

  extractUniqueCurrentLevels(): void {
    if (this.manager && this.manager.team) {
      const allLevels = this.manager.team.flatMap(member =>
        member.skills.map(skill => skill.current_expertise).filter(level => level)
      );
      this.uniqueCurrentLevels = [...new Set(allLevels)];
    }
  }

  assignTraining(): void {
    if (!this.selectedTrainingIds || this.selectedTrainingIds.length === 0) {
      this.toastService.warning('Please select at least one training module.');
      return;
    }
    if (!this.selectedMemberIds || this.selectedMemberIds.length === 0) {
      this.toastService.warning('Please select at least one team member.');
      return;
    }
    
    // Filter out already assigned combinations - only process valid assignments
    const validAssignments: { trainingId: number; memberId: string; trainingName: string; memberName: string }[] = [];
    
    this.selectedTrainingIds.forEach(trainingId => {
      const training = this.trainingCatalog.find(t => t.id === trainingId);
      const trainingName = training?.training_name || 'Unknown Training';
      
      this.selectedMemberIds.forEach(memberId => {
        const member = this.manager?.team.find(m => m.id === memberId);
        const memberName = member?.name || 'Unknown';
        
        // Only add if not already assigned
        if (!this.isAlreadyAssigned(trainingId, memberId)) {
          validAssignments.push({ trainingId, memberId, trainingName, memberName });
        }
      });
    });
    
    // If no valid assignments after filtering duplicates
    if (validAssignments.length === 0) {
      this.toastService.warning('All selected assignments already exist. Please select different training or team members.');
      return;
    }
    
    // Proceed with valid assignments only
    const trainingNames: string[] = [];
    const memberNames: string[] = [];
    const failedAssignments: { training: string; member: string }[] = [];
    let completedCount = 0;

    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication token missing. Please login again.');
      return;
    }
    
    this.isAssigningTraining = true;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    // Create array of observables only for valid (non-duplicate) assignments
    const assignmentObservables: any[] = [];
    
    validAssignments.forEach(({ trainingId, memberId, trainingName, memberName }) => {
      if (!trainingNames.includes(trainingName)) {
        trainingNames.push(trainingName);
      }
      if (!memberNames.includes(memberName)) {
        memberNames.push(memberName);
      }
      
      const payload = { training_id: trainingId, employee_username: memberId };
      assignmentObservables.push(
        this.http.post(this.apiService.assignmentsUrl, payload, { headers }).pipe(
          map(() => {
            // Add to existing assignments to prevent immediate re-assignment
            const key = `${trainingId}_${memberId}`;
            this.existingAssignments.add(key);
            return { success: true, trainingName, memberName };
          }),
          catchError((err) => {
            // Check if it's a duplicate error (400)
            if (err.status === 400 && err.error?.detail?.includes('already assigned')) {
              const key = `${trainingId}_${memberId}`;
              this.existingAssignments.add(key);
            }
            return of({ success: false, trainingName, memberName, error: err });
          })
        )
      );
    });

    // Execute all assignments in parallel using forkJoin
    forkJoin(assignmentObservables).subscribe({
      next: (results) => {
        this.isAssigningTraining = false;
        
        // Count successful assignments
        const successfulNames: string[] = [];
        results.forEach((result: any) => {
          if (result.success) {
            completedCount++;
            if (!successfulNames.includes(result.memberName)) {
              successfulNames.push(result.memberName);
            }
          } else {
            failedAssignments.push({ training: result.trainingName, member: result.memberName });
          }
        });
        
        if (completedCount > 0) {
          // Refresh team assigned trainings to update the list
          this.fetchTeamAssignedTrainings();
          
          // Show success modal
          this.assignmentSuccessData = {
            trainingNames: [...new Set(trainingNames)],
            memberNames: successfulNames,
            totalAssignments: completedCount
          };
          this.showAssignmentSuccessModal = true;
          
          // Reset selections
          this.selectedTrainingIds = [];
          this.selectedMemberIds = [];
          
          // Show toasts for results
          // Note: duplicateAssignments info is already shown in the modal
          if (failedAssignments.length > 0) {
            this.toastService.warning(`${failedAssignments.length} assignment(s) failed. Please try again.`);
          } else {
            this.toastService.success(`Successfully assigned ${completedCount} training(s)!`);
          }
        } else {
          const msg = failedAssignments.length > 0 ? 'All assignments failed. Please try again.' : 'Failed to assign training.';
          this.toastService.error(msg);
        }
      },
      error: (err) => {
        this.isAssigningTraining = false;
        console.error('Error assigning trainings:', err);
        this.toastService.error('Failed to assign trainings. Please try again.');
      }
    });
  }
  
  closeDuplicateModal(): void {
    this.showDuplicateModal = false;
    this.duplicateAssignments = [];
    this.pendingValidAssignments = [];
  }
  
  confirmProceedWithAssignments(): void {
    this.showDuplicateModal = false;
    const validAssignments = this.pendingValidAssignments;
    this.duplicateAssignments = [];
    this.pendingValidAssignments = [];
    
    if (validAssignments.length === 0) {
      this.toastService.warning('No valid assignments to proceed with.');
      return;
    }
    
    // Process assignments inline (duplicate modal is kept for backward compatibility but not used in new flow)
    // This method is only called if the duplicate modal is shown, which shouldn't happen with the new inline indicators
    // But keeping it for safety
    this.toastService.info('Please use the inline indicators to see duplicate assignments. This modal is deprecated.');
  }

  toggleTrainingSelection(trainingId: number): void {
    const index = this.selectedTrainingIds.indexOf(trainingId);
    if (index > -1) {
      this.selectedTrainingIds.splice(index, 1);
    } else {
      this.selectedTrainingIds.push(trainingId);
    }
  }
  
  isAlreadyAssigned(trainingId: number, memberId: string): boolean {
    const assignmentKey = `${trainingId}_${memberId}`;
    return this.existingAssignments.has(assignmentKey);
  }
  
  getAlreadyAssignedMembersForTraining(trainingId: number): string[] {
    const assignedMembers: string[] = [];
    this.manager?.team.forEach(member => {
      if (this.isAlreadyAssigned(trainingId, member.id)) {
        assignedMembers.push(member.name);
      }
    });
    return assignedMembers;
  }
  
  getAlreadyAssignedTrainingsForMember(memberId: string): string[] {
    const assignedTrainings: string[] = [];
    this.trainingCatalog.forEach(training => {
      if (this.isAlreadyAssigned(training.id, memberId)) {
        assignedTrainings.push(training.training_name);
      }
    });
    return assignedTrainings;
  }
  
  getSelectedTrainingsAlreadyAssignedForMember(memberId: string): string[] {
    const alreadyAssigned: string[] = [];
    this.selectedTrainingIds.forEach(trainingId => {
      if (this.isAlreadyAssigned(trainingId, memberId)) {
        const training = this.trainingCatalog.find(t => t.id === trainingId);
        if (training) {
          alreadyAssigned.push(training.training_name);
        }
      }
    });
    return alreadyAssigned;
  }
  
  canAssignTrainingToMember(trainingId: number, memberId: string): boolean {
    return !this.isAlreadyAssigned(trainingId, memberId);
  }
  
  getSelectedMembersAlreadyAssigned(trainingId: number): string[] {
    const alreadyAssigned: string[] = [];
    this.selectedMemberIds.forEach(memberId => {
      if (this.isAlreadyAssigned(trainingId, memberId)) {
        const member = this.manager?.team.find(m => m.id === memberId);
        if (member) {
          alreadyAssigned.push(member.name);
        }
      }
    });
    return alreadyAssigned;
  }

  isTrainingSelected(trainingId: number): boolean {
    return this.selectedTrainingIds.includes(trainingId);
  }

  selectAllTrainings(): void {
    this.selectedTrainingIds = this.filteredAssignTrainings.map(t => t.id);
  }

  clearAllTrainings(): void {
    this.selectedTrainingIds = [];
  }

  toggleMemberSelection(memberId: string): void {
    const index = this.selectedMemberIds.indexOf(memberId);
    if (index > -1) {
      this.selectedMemberIds.splice(index, 1);
    } else {
      this.selectedMemberIds.push(memberId);
    }
  }

  isMemberSelected(memberId: string): boolean {
    return this.selectedMemberIds.includes(memberId);
  }

  closeAssignmentSuccessModal(): void {
    this.showAssignmentSuccessModal = false;
    this.assignmentSuccessData = null;
  }

  selectAllMembers(): void {
    if (this.manager && this.manager.team) {
      this.selectedMemberIds = this.filteredAssignMembers.map(m => m.id);
    }
  }

  clearAllMembers(): void {
    this.selectedMemberIds = [];
  }

  getMemberNameById(memberId: string): string {
    if (!this.manager || !this.manager.team) return '';
    const member = this.manager.team.find(m => m.id === memberId);
    return member?.name || '';
  }

  getSelectedTrainingNames(): string {
    if (!this.selectedTrainingIds || this.selectedTrainingIds.length === 0) return '...';
    if (this.selectedTrainingIds.length === 1) {
      return this.trainingCatalog.find(t => t.id === this.selectedTrainingIds[0])?.training_name || '...';
    }
    return `${this.selectedTrainingIds.length} Trainings Selected`;
  }

  getTrainingNameById(trainingId: number): string {
    return this.trainingCatalog.find(t => t.id === trainingId)?.training_name || '';
  }

  getSelectedMemberNames(): string {
    if (!this.selectedMemberIds || this.selectedMemberIds.length === 0 || !this.manager) return '...';
    if (this.selectedMemberIds.length === 1) {
      return this.manager.team.find(m => m.id === this.selectedMemberIds[0])?.name || '...';
    }
    return `${this.selectedMemberIds.length} Members Selected`;
  }

  calculateProgress(skills: Competency[]): number {
    const total = skills.length;
    const met = skills.filter(s => s.status === 'Met').length;
    return total > 0 ? (met / total) * 100 : 0;
  }

  getTeamProgressPercentage(): number {
    const totalSkills = this.manager?.team.flatMap(member => member.skills).length || 0;
    const metSkills = this.manager?.team.flatMap(member => member.skills).filter(skill => skill.status === 'Met').length || 0;
    return totalSkills > 0 ? Math.round((metSkills / totalSkills) * 100) : 0;
  }

  getMySkillsMetCount(): number {
    return this.manager?.skills.filter(skill => skill.status === 'Met').length || 0;
  }

  // Helper method to get feedback for a specific skill (similar to engineer dashboard)
  getFeedbackForSkill(skillName: string, employeeEmpid?: string): ManagerPerformanceFeedback[] {
    if (!skillName || !skillName.trim()) {
      return [];
    }

    // If no trainings have been loaded yet, return empty array
    if (!this.allTrainings || this.allTrainings.length === 0) {
      return [];
    }

    // Find trainings that match this skill (case-insensitive, trimmed)
    const skillNameLower = skillName.toLowerCase().trim();
    const matchingTrainings = this.allTrainings.filter(t => {
      if (!t || !t.skill) return false;
      return t.skill.toLowerCase().trim() === skillNameLower;
    });

    if (matchingTrainings.length === 0) {
      return [];
    }

    // Get training IDs for matching trainings
    const trainingIds = matchingTrainings.map(t => t.id).filter(id => id != null);
    if (trainingIds.length === 0) {
      return [];
    }

    // Get feedback from feedbackHistory (which contains all feedback for the selected employee)
    // If employeeEmpid is provided, filter by it; otherwise, this is for manager's own skills
    // For manager's own skills, we might not have feedback, so return empty array
    if (!employeeEmpid) {
      // Manager's own skills - feedback might not be available in the same way
      // Return empty array for now, can be enhanced later if needed
      return [];
    }

    // Filter feedback history by training IDs and employee
    const filteredFeedback = this.feedbackHistory.filter(feedback => {
      if (!feedback || !feedback.training_id) return false;
      if (feedback.employee_empid !== employeeEmpid) return false;
      return trainingIds.includes(feedback.training_id);
    });

    // Enrich with training info
    const trainingMap = new Map<number, any>();
    matchingTrainings.forEach(t => {
      if (t.id != null) {
        trainingMap.set(t.id, t);
      }
    });

    return filteredFeedback.map(feedback => {
      const training = trainingMap.get(feedback.training_id);
      return {
        ...feedback,
        skill_category: training?.skill_category || null,
        training_name: training?.training_name || feedback.training_name
      } as ManagerPerformanceFeedback;
    });
  }

  getSkillProgress(skill: Competency): number {
    // Calculate progress based on average overall performance across L1-L5 levels
    const skillName = skill.skill;
    if (!skillName) return 0;

    // For manager dashboard, we need employee_empid to get feedback
    // Since this is used for manager's own skills, we might not have feedback
    // Try to get feedback - if not available, fall back to old calculation
    const feedbackList = this.getFeedbackForSkill(skillName);
    
    // If no feedback available, fall back to old calculation
    if (!feedbackList || feedbackList.length === 0) {
      const extractLevel = (level: string): number => {
        if (!level) return 0;
        if (level.toUpperCase().startsWith('L')) {
          return parseInt(level.substring(1), 10) || 0;
        }
        const levelMap: { [key: string]: number } = {
          'BEGINNER': 1, 'INTERMEDIATE': 2, 'ADVANCED': 3, 'EXPERT': 4
        };
        return levelMap[level.toUpperCase()] || 0;
      };

      const current = extractLevel(skill.current_expertise);
      const target = extractLevel(skill.target_expertise);

      if (target === 0) return 0;

      let percent = Math.round((current / target) * 100);
      if (percent > 100) percent = 100;
      if (percent < 0) percent = 0;
      return percent;
    }

    // Group feedback by skill category (L1, L2, L3, L4, L5)
    const feedbackByLevel = new Map<string, ManagerPerformanceFeedback[]>();
    feedbackList.forEach(feedback => {
      const skillCategory = feedback.skill_category || 'Unknown';
      if (skillCategory.startsWith('L') && ['L1', 'L2', 'L3', 'L4', 'L5'].includes(skillCategory)) {
        if (!feedbackByLevel.has(skillCategory)) {
          feedbackByLevel.set(skillCategory, []);
        }
        feedbackByLevel.get(skillCategory)!.push(feedback);
      }
    });

    // Sort feedback within each level by updated_at (most recent first)
    feedbackByLevel.forEach((feedbackList, level) => {
      feedbackList.sort((a, b) => {
        const dateA = new Date(a.updated_at || a.created_at || 0).getTime();
        const dateB = new Date(b.updated_at || b.created_at || 0).getTime();
        return dateB - dateA;
      });
    });

    // Get latest overall_performance for each level (L1-L5)
    const performanceScores: number[] = [];
    const levels = ['L1', 'L2', 'L3', 'L4', 'L5'];
    
    levels.forEach(level => {
      const levelFeedback = feedbackByLevel.get(level);
      if (levelFeedback && levelFeedback.length > 0) {
        const latestFeedback = levelFeedback[0];
        if (latestFeedback.overall_performance != null && latestFeedback.overall_performance > 0) {
          performanceScores.push(latestFeedback.overall_performance);
        }
      }
    });

    // If no performance scores found, fall back to old calculation
    if (performanceScores.length === 0) {
      const extractLevel = (level: string): number => {
        if (!level) return 0;
        if (level.toUpperCase().startsWith('L')) {
          return parseInt(level.substring(1), 10) || 0;
        }
        const levelMap: { [key: string]: number } = {
          'BEGINNER': 1, 'INTERMEDIATE': 2, 'ADVANCED': 3, 'EXPERT': 4
        };
        return levelMap[level.toUpperCase()] || 0;
      };

      const current = extractLevel(skill.current_expertise);
      const target = extractLevel(skill.target_expertise);

      if (target === 0) return 0;

      let percent = Math.round((current / target) * 100);
      if (percent > 100) percent = 100;
      if (percent < 0) percent = 0;
      return percent;
    }

    // Calculate average of overall performance scores
    const sum = performanceScores.reduce((acc, score) => acc + score, 0);
    const average = sum / performanceScores.length;

    // Convert to percentage: (average/5) * 100
    // Since overall_performance is on a 1-5 scale, we convert it to percentage
    let percent = Math.round((average / 5) * 100);
    if (percent > 100) percent = 100;
    if (percent < 0) percent = 0;
    return percent;
  }

  getFilteredMySkills(): Competency[] {
    if (!this.manager || !this.manager.skills) {
      return [];
    }
    let filtered = [...this.manager.skills];
    if (this.mySkillsSearch) {
      filtered = filtered.filter(skill =>
        skill.skill.toLowerCase().includes(this.mySkillsSearch.toLowerCase())
      );
    }
    if (this.mySkillsStatusFilter !== 'All') {
      filtered = filtered.filter(skill => skill.status === this.mySkillsStatusFilter);
    }
    if (this.mySkillsSkillFilter !== 'All') {
      filtered = filtered.filter(skill => skill.skill === this.mySkillsSkillFilter);
    }
    return filtered;
  }

  getTeamSkillsMetCount(member: TeamMember): number {
    return member.skills.filter(s => s.status === 'Met').length;
  }

  getTeamSkillsGapCount(member: TeamMember): number {
    return member.skills.filter(s => s.status === 'Gap').length;
  }

  getMemberProgressPercentage(member: TeamMember): number {
    return this.calculateProgress(member.skills);
  }

  getTotalTeamSkillsCount(): number {
    return (this.manager?.team || []).reduce((sum, m) => sum + m.skills.length, 0);
  }

  viewMemberDetails(member: TeamMember): void {
    this.selectedTeamMember = member;
  }

  clearMemberDetails(): void {
    this.selectedTeamMember = null;
    this.cancelEditSkill();
  }

  logout(): void {
    // Clear notifications before logging out
    this.notificationService.clear();
    this.authService.logout();
    this.router.navigate(['/login']);
  }

  get filteredTeamMembers(): TeamMember[] {
    if (!this.manager || !this.manager.team) {
      return [];
    }
    let filteredMembers = [...this.manager.team];
    if (this.teamMemberNameFilter !== 'All') {
      filteredMembers = filteredMembers.filter(member => member.name === this.teamMemberNameFilter);
    }
    if (this.teamSkillsStatusFilter !== 'All' || this.teamSkillsSkillFilter !== 'All' || this.teamCompetencyFilter !== 'All' || this.teamSkillsCurrentLevelFilter !== 'All') {
      filteredMembers = filteredMembers.filter(member => {
        return member.skills.some(skill => {
          const statusMatch = this.teamSkillsStatusFilter === 'All' || skill.status === this.teamSkillsStatusFilter;
          const skillMatch = this.teamSkillsSkillFilter === 'All' || skill.skill === this.teamSkillsSkillFilter;
          const competencyMatch = this.teamCompetencyFilter === 'All' || skill.competency === this.teamCompetencyFilter;
          const levelMatch = this.teamSkillsCurrentLevelFilter === 'All' || skill.current_expertise === this.teamSkillsCurrentLevelFilter;
          return statusMatch && skillMatch && competencyMatch && levelMatch;
        });
      });
    }
    return filteredMembers;
  }

  get filteredAssignTrainings(): TrainingDetail[] {
    let list = [...(this.trainingCatalog || [])];
    const q = (this.assignTrainingSearch || '').trim().toLowerCase();
    if (q) {
      list = list.filter(t =>
        (t.training_name || '').toLowerCase().includes(q) ||
        (t.training_topics || '').toLowerCase().includes(q) ||
        (t.trainer_name || '').toLowerCase().includes(q) ||
        (t.training_type || '').toLowerCase().includes(q)
      );
    }
    return list;
  }

  get filteredAssignedTrainings(): TrainingDetail[] {
    let list = [...(this.assignedTrainings || [])];
    if (this.assignedSearch && this.assignedSearch.trim()) {
      const q = this.assignedSearch.trim().toLowerCase();
      list = list.filter(t =>
        (t.training_name || '').toLowerCase().includes(q) ||
        (t.trainer_name || '').toLowerCase().includes(q) ||
        (t.skill || '').toLowerCase().includes(q)
      );
    }
    if (this.assignedSkillFilter !== 'All') {
      list = list.filter(t => t.skill === this.assignedSkillFilter);
    }
    if (this.assignedLevelFilter !== 'All') {
      list = list.filter(t => t.skill_category === this.assignedLevelFilter);
    }
    if (this.assignedDateFilter) {
      list = list.filter(t => t.training_date === this.assignedDateFilter);
    }
    list.sort((a, b) => {
      const dateA = a.training_date ? new Date(a.training_date).getTime() : Infinity;
      const dateB = b.training_date ? new Date(b.training_date).getTime() : Infinity;
      return dateA - dateB;
    });
    return list;
  }

  get filteredAssignMembers(): TeamMember[] {
    if (!this.manager || !this.manager.team) return [];
    const q = (this.assignMemberSearch || '').trim().toLowerCase();
    if (!q) return this.manager.team;
    return this.manager.team.filter(m => (m.name || '').toLowerCase().includes(q));
  }

  startEditSkill(memberId: string, skillIndex: number): void {
    const member = this.manager?.team.find(m => m.id === memberId);
    if (member && member.skills[skillIndex]) {
      this.editingSkill = { memberId, skillIndex };
      this.editSkillData = {
        current_expertise: member.skills[skillIndex].current_expertise,
        target_expertise: member.skills[skillIndex].target_expertise
      };
      this.errorMessage = '';
      this.successMessage = '';
    }
  }

  cancelEditSkill(): void {
    this.editingSkill = null;
    this.editSkillData = { current_expertise: '', target_expertise: '' };
  }

  saveSkillEdit(): void {
    if (!this.editingSkill || !this.manager) return;
    const member = this.manager.team.find(m => m.id === this.editingSkill!.memberId);
    if (member && member.skills[this.editingSkill.skillIndex]) {
      const skill = member.skills[this.editingSkill.skillIndex];
      const updateRequest = {
        employee_username: member.id,
        skill_name: skill.skill,
        current_expertise: this.editSkillData.current_expertise,
        target_expertise: this.editSkillData.target_expertise
      };
      const token = this.authService.getToken();
      if (!token) {
        this.errorMessage = 'Authentication token not found. Please login again.';
        return;
      }
      const headers = new HttpHeaders({
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      });
      this.http.put(this.apiService.managerTeamSkillUpdateUrl, updateRequest, { headers })
        .subscribe({
          next: (response: any) => {
            skill.current_expertise = response.current_expertise;
            skill.target_expertise = response.target_expertise;
            skill.status = response.status;
            this.processDashboardData();
            this.cancelEditSkill();
            this.successMessage = `Skill "${skill.skill}" updated successfully!`;
            setTimeout(() => { this.successMessage = ''; }, 3000);
          },
          error: (error) => {
            this.errorMessage = 'Failed to update skill. Please try again.';
          }
        });
    }
  }

  isEditingSkill(memberId: string, skillIndex: number): boolean {
    return this.editingSkill?.memberId === memberId && this.editingSkill?.skillIndex === skillIndex;
  }

  // Additional Skills Management
  loadAdditionalSkills(): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get<any[]>(this.apiService.additionalSkillsUrl, { headers }).subscribe({
      next: (skills) => { this.additionalSkills = skills; },
      error: (err) => { this.additionalSkills = []; }
    });
  }

  toggleAddSkillForm(): void {
    this.showAddSkillForm = !this.showAddSkillForm;
    if (!this.showAddSkillForm) {
      this.resetNewSkillForm();
    }
  }

  addNewSkill(): void {
    if (this.newSkill.name.trim()) {
      const token = this.authService.getToken();
      if (!token) return;
      const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
      const skillData = {
        skill_name: this.newSkill.name.trim(),
        skill_level: this.newSkill.level,
        skill_category: this.newSkill.category,
        description: this.newSkill.description.trim() || null
      };
      if (this.editingSkillId) {
        this.http.put<any>(this.apiService.additionalSkillUrl(this.editingSkillId), skillData, { headers }).subscribe({
          next: (updatedSkill) => {
            const index = this.additionalSkills.findIndex(s => s.id === this.editingSkillId);
            if (index !== -1) { this.additionalSkills[index] = updatedSkill; }
            this.resetNewSkillForm();
            this.showAddSkillForm = false;
          },
          error: (err) => { console.error('Failed to update skill:', err); }
        });
      } else {
        this.http.post<any>(this.apiService.additionalSkillsUrl, skillData, { headers }).subscribe({
          next: (newSkill) => {
            this.additionalSkills.push(newSkill);
            this.resetNewSkillForm();
            this.showAddSkillForm = false;
          },
          error: (err) => { console.error('Failed to add skill:', err); }
        });
      }
    }
  }

  removeAdditionalSkill(skillId: number): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.delete(this.apiService.additionalSkillUrl(skillId), { headers }).subscribe({
      next: () => { this.additionalSkills = this.additionalSkills.filter(skill => skill.id !== skillId); },
      error: (err) => { console.error('Failed to delete skill:', err); }
    });
  }

  editAdditionalSkill(skill: any): void {
    this.newSkill = {
      name: skill.skill_name,
      level: skill.skill_level,
      category: skill.skill_category,
      description: skill.description || ''
    };
    this.showAddSkillForm = true;
    this.editingSkillId = skill.id;
  }

  resetNewSkillForm(): void {
    this.newSkill = { name: '', level: 'Beginner', category: 'Technical', description: '' };
    this.editingSkillId = null;
  }

  getSkillLevelColor(level: string): string {
    switch (level) {
      case 'Expert': return 'bg-purple-100 text-purple-800';
      case 'Advanced': return 'bg-blue-100 text-blue-800';
      case 'Intermediate': return 'bg-green-100 text-green-800';
      case 'Beginner': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  }

  getCategoryColor(category: string): string {
    switch (category) {
      case 'Technical': return 'bg-slate-100 text-slate-700 border border-slate-300';
      case 'Soft Skills': return 'bg-stone-100 text-stone-700 border border-stone-300';
      case 'Leadership': return 'bg-zinc-100 text-zinc-700 border border-zinc-300';
      case 'Communication': return 'bg-neutral-100 text-neutral-700 border border-neutral-300';
      case 'Project Management': return 'bg-gray-100 text-gray-700 border border-gray-300';
      default: return 'bg-gray-100 text-gray-700 border border-gray-300';
    }
  }
  
  // Levels helper methods
  getFilteredSections(): Section[] {
    let sectionsToFilter = this.sections;
    if (this.selectedSkill) {
      sectionsToFilter = this.sections.filter((sec: any) => sec.title === this.selectedSkill);
    }
    const q = this.levelsSearch.trim().toLowerCase();
    if (!q) return sectionsToFilter;

    return sectionsToFilter.map((sec: any) => {
      const matchTitle = sec.title.toLowerCase().includes(q) || (sec.subtitle ?? '').toLowerCase().includes(q);
      const filteredLevels = sec.levels
        .map((l: any) => ({ ...l, items: l.items.filter((it: string) => it.toLowerCase().includes(q)) }))
        .filter((l: any) => l.items.length > 0 || `level ${l.level}`.includes(q));
      if (matchTitle || filteredLevels.length) {
        return { ...sec, levels: matchTitle ? sec.levels : filteredLevels };
      }
      return null;
    }).filter((s: any) => s !== null);
  }

  getLevelHeaderClass = (level: number) => ['bg-red-50', 'bg-orange-50', 'bg-yellow-50', 'bg-indigo-50', 'bg-green-50'][level - 1] || 'bg-gray-50';
  getLevelBadgeClass = (level: number) => ['bg-indigo-500', 'bg-indigo-500', 'bg-indigo-500', 'bg-indigo-500', 'bg-indigo-500'][level - 1] || 'bg-gray-500';
  getLevelTitle = (level: number) => ['Beginner', 'Basic', 'Intermediate', 'Advanced', 'Expert'][level - 1] || 'Unknown';
  getLevelIcon = (level: number) => ['fa-solid fa-seedling text-indigo-500', 'fa-solid fa-leaf text-indigo-500', 'fa-solid fa-tree text-indigo-600', 'fa-solid fa-rocket text-indigo-500', 'fa-solid fa-crown text-indigo-500'][level - 1] || 'fa-solid fa-circle';
  getComplexityDots = (level: number) => Array.from({ length: 5 }, (_, i) => i < level);
  onSkillChange(): void {}

  // --- Levels Tab Helpers ---
  public getLevelKey(sectionTitle: string, level: number): string {
    return `${sectionTitle}-${level}`;
  }

  public toggleLevelExpansion(key: string): void {
    if (this.expandedLevels.has(key)) {
      this.expandedLevels.delete(key);
    } else {
      this.expandedLevels.add(key);
    }
  }

  public isLevelExpanded(key: string): boolean {
    return this.expandedLevels.has(key);
  }

  // <<< NEW METHOD FOR ACCORDION >>>
  public toggleSkillExpansion(skillTitle: string): void {
    if (this.expandedSkill === skillTitle) {
        this.expandedSkill = null; // Collapse if clicking the same one again
    } else {
        this.expandedSkill = skillTitle; // Expand the new one
    }
  }

  public getLevelItems(section: Section, levelNum: number): string[] {
    const levelData = section.levels.find(l => l.level === levelNum);
    return levelData ? levelData.items : [];
  }

  // --- Filter Reset Logic ---
  resetMySkillsFilters(): void {
    this.mySkillsSearch = '';
    this.mySkillsSkillFilter = 'All';
    this.mySkillsStatusFilter = 'All';
  }

  resetCatalogFilters(): void {
    this.catalogSearch = '';
    this.catalogTypeFilter = 'All';
    this.catalogCategoryFilter = 'All';
  }

  resetAssignedTrainingFilters(): void {
    this.assignedSearch = '';
    this.assignedSkillFilter = 'All';
    this.assignedLevelFilter = 'All';
    this.assignedDateFilter = '';
  }

  // --- View Toggle Logic ---
  setTrainingCatalogView(view: 'list' | 'calendar'): void {
    this.trainingCatalogView = view;
    if (view === 'calendar') {
      this.generateCalendar();
    }
  }

  setAssignedTrainingsView(view: 'list' | 'calendar'): void {
    this.assignedTrainingsView = view;
    if (view === 'calendar') {
      this.generateCalendar();
    }
  }

  // --- Trainer Zone Methods ---
  setTrainerZoneView(view: 'overview' | 'assignmentForm' | 'feedbackForm'): void {
    if (view === 'overview') {
      this.resetNewAssignmentForm();
      this.resetNewFeedbackForm();
    }
    this.trainerZoneView = view;
  }

  isAssignmentShared(trainingId: number): boolean {
    return this.sharedAssignments.get(trainingId) || false;
  }

  isFeedbackShared(trainingId: number): boolean {
    return this.sharedFeedback.get(trainingId) || false;
  }

  checkSharedStatus(trainingId: number): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    // Check if assignment is shared (for trainers)
    this.http.get(this.apiService.trainerAssignmentsUrl(trainingId), { headers }).subscribe({
      next: (response: any) => {
        if (response) {
          this.sharedAssignments.set(trainingId, true);
          // Store who shared it
          if (response.trainer_username) {
            this.assignmentSharedBy.set(trainingId, response.trainer_username);
          }
        } else {
          this.sharedAssignments.set(trainingId, false);
          this.assignmentSharedBy.delete(trainingId);
        }
      },
      error: () => {
        this.sharedAssignments.set(trainingId, false);
        this.assignmentSharedBy.delete(trainingId);
      }
    });
    
    // Check if feedback is shared (for trainers)
    this.http.get(this.apiService.trainerFeedbackUrl(trainingId), { headers }).subscribe({
      next: (response: any) => {
        if (response) {
          this.sharedFeedback.set(trainingId, true);
          // Store who shared it
          if (response.trainer_username) {
            this.feedbackSharedBy.set(trainingId, response.trainer_username);
          }
        } else {
          this.sharedFeedback.set(trainingId, false);
          this.feedbackSharedBy.delete(trainingId);
        }
      },
      error: () => {
        this.sharedFeedback.set(trainingId, false);
        this.feedbackSharedBy.delete(trainingId);
      }
    });
  }

  fetchTrainingCandidates(trainingId: number): void {
    const token = this.authService.getToken();
    if (!token) {
      this.toastService.warning('Authentication token missing. Please login again.');
      return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.http.get<{employee_empid: string, employee_name: string, attended: boolean}[]>(
      this.apiService.getTrainingCandidatesUrl(trainingId), 
      { headers }
    ).subscribe({
      next: (candidates) => {
        this.trainingCandidates.set(trainingId, candidates);
        if (candidates.length === 0) {
          this.toastService.info('No candidates assigned to this training yet.');
        }
      },
      error: (err) => {
        // If 401, token expired - redirect to login
        if (err.status === 401) {
          this.toastService.error('Your session has expired. Please log in again.');
          this.authService.logout();
          this.router.navigate(['/login']);
        } else if (err.status === 403) {
          // If 403, user is not the trainer
          this.toastService.warning('Only the trainer of this training can view and mark attendance.');
        } else {
          this.toastService.error('Failed to fetch training candidates. Please try again.');
          console.error('Failed to fetch training candidates:', err);
        }
        this.trainingCandidates.set(trainingId, []);
      }
    });
  }

  getTrainingCandidates(trainingId: number): {employee_empid: string, employee_name: string, attended: boolean}[] {
    return this.trainingCandidates.get(trainingId) || [];
  }

  openAttendanceModal(trainingId: number): void {
    this.selectedTrainingForAttendance = trainingId;
    const candidates = this.getTrainingCandidates(trainingId);
    
    // Check if candidates list is empty (user might not be trainer or candidates not loaded)
    if (candidates.length === 0) {
      // Try to fetch candidates first
      this.fetchTrainingCandidates(trainingId);
      
      // Wait a bit and check again, or show warning
      setTimeout(() => {
        const updatedCandidates = this.getTrainingCandidates(trainingId);
        if (updatedCandidates.length === 0) {
          this.toastService.warning('No candidates found for this training. You may not be the trainer for this training, or no employees have been assigned yet.');
          return;
        }
        this.attendanceCandidates = updatedCandidates.map(c => ({ ...c }));
        this.showAttendanceModal = true;
      }, 500);
      return;
    }
    
    // Create a copy for editing
    this.attendanceCandidates = candidates.map(c => ({ ...c }));
    this.showAttendanceModal = true;
  }

  closeAttendanceModal(): void {
    this.showAttendanceModal = false;
    this.selectedTrainingForAttendance = null;
    this.attendanceCandidates = [];
  }

  toggleCandidateAttendance(index: number): void {
    this.attendanceCandidates[index].attended = !this.attendanceCandidates[index].attended;
  }

  getAttendedCount(): number {
    return this.attendanceCandidates.filter(c => c.attended).length;
  }

  getTotalCandidatesCount(): number {
    return this.attendanceCandidates.length;
  }

  canMarkAttendance(): boolean {
    return !this.isMarkingAttendance && this.getAttendedCount() > 0;
  }

  markAttendance(): void {
    if (!this.selectedTrainingForAttendance) return;
    
    const token = this.authService.getToken();
    if (!token) {
      this.toastService.warning('Authentication token missing. Please login again.');
      return;
    }
    
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const attendedEmpids = this.attendanceCandidates
      .filter(c => c.attended)
      .map(c => c.employee_empid);
    
    this.isMarkingAttendance = true;
    
    // Get attended candidate names for better success message
    const attendedNames = this.attendanceCandidates
      .filter(c => c.attended)
      .map(c => c.employee_name)
      .join(', ');
    
    this.http.post(
      this.apiService.markTrainingAttendanceUrl(this.selectedTrainingForAttendance),
      { candidate_empids: attendedEmpids },
      { headers }
    ).subscribe({
      next: (response: any) => {
        // Show immediate success popup with detailed information
        // Find the training object from the ID
        const training = this.allTrainings.find(t => t.id === this.selectedTrainingForAttendance);
        const trainingName = training?.training_name || 'Training';
        const attendedCount = response.attended_count || attendedEmpids.length;
        const totalCount = response.total_assigned || this.attendanceCandidates.length;
        const absentCount = totalCount - attendedCount;
        
        // Prepare data for success popup
        this.attendanceSuccessData = {
          trainingName: trainingName,
          attendedCount: attendedCount,
          absentCount: absentCount,
          totalCount: totalCount,
          attendedNames: attendedNames || ''
        };
        
        // Close attendance modal first
        this.closeAttendanceModal();
        
        // Show success popup in the center
        this.showAttendanceSuccessPopup = true;
        
        // Refresh candidates list immediately
        this.fetchTrainingCandidates(this.selectedTrainingForAttendance!);
      },
      error: (err) => {
        console.error('Failed to mark attendance:', err);
        if (err.status === 401) {
          this.toastService.error('Your session has expired. Please log in again.');
          this.authService.logout();
          this.router.navigate(['/login']);
        } else if (err.status === 403) {
          this.toastService.warning('Only the trainer of this training can mark attendance.');
        } else if (err.status === 400) {
          this.toastService.warning(err.error?.detail || 'Invalid request. Please check the selected candidates.');
        } else {
          this.toastService.warning('Failed to mark attendance. Please try again.');
        }
      },
      complete: () => {
        this.isMarkingAttendance = false;
      }
    });
  }

  getAssignmentSharedBy(trainingId: number): string {
    return this.assignmentSharedBy.get(trainingId) || '';
  }

  getFeedbackSharedBy(trainingId: number): string {
    return this.feedbackSharedBy.get(trainingId) || '';
  }

  openShareAssignment(trainingId: number): void {
    this.resetNewAssignmentForm();
    this.newAssignment.trainingId = trainingId;
    
    // Check shared status and load existing data if available
    const token = this.authService.getToken();
    if (token) {
      const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
      this.http.get(this.apiService.trainerAssignmentsUrl(trainingId), { headers }).subscribe({
        next: (response: any) => {
          if (response) {
            // Assignment already exists - load it for editing
            this.sharedAssignments.set(trainingId, true);
            if (response.trainer_username) {
              this.assignmentSharedBy.set(trainingId, response.trainer_username);
            }
            
            // Load existing assignment data
            this.newAssignment.title = response.title || '';
            this.newAssignment.description = response.description || '';
            this.newAssignment.questions = response.questions || [];
            
            const currentUser = this.authService.getUsername() || this.manager?.id || '';
            if (response.trainer_username && response.trainer_username !== currentUser) {
              this.toastService.info(`Assignment already shared by your co-trainer (${response.trainer_username}). You can update it below.`);
            } else {
              this.toastService.info('Loading existing assignment for editing.');
            }
          } else {
            this.sharedAssignments.set(trainingId, false);
            this.assignmentSharedBy.delete(trainingId);
          }
          this.setTrainerZoneView('assignmentForm');
        },
        error: () => {
          this.sharedAssignments.set(trainingId, false);
          this.assignmentSharedBy.delete(trainingId);
          this.setTrainerZoneView('assignmentForm');
        }
      });
    } else {
      this.setTrainerZoneView('assignmentForm');
    }
  }

  openShareFeedback(trainingId: number): void {
    this.resetNewFeedbackForm();
    this.newFeedback.trainingId = trainingId;
    
    // Check shared status and load existing data if available
    const token = this.authService.getToken();
    if (token) {
      const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
      this.http.get(this.apiService.trainerFeedbackUrl(trainingId), { headers }).subscribe({
        next: (response: any) => {
          if (response) {
            // Feedback already exists - load it for editing
            this.sharedFeedback.set(trainingId, true);
            if (response.trainer_username) {
              this.feedbackSharedBy.set(trainingId, response.trainer_username);
            }
            
            // Load existing feedback data
            this.newFeedback.customQuestions = (response.customQuestions || []).map((q: any) => ({
              text: q.text || '',
              options: q.options || [],
              isDefault: q.isDefault || false
            }));
            
            const currentUser = this.authService.getUsername() || this.manager?.id || '';
            if (response.trainer_username && response.trainer_username !== currentUser) {
              this.toastService.info(`Feedback has already been shared by your co-trainer (${response.trainer_username}). You can update it below.`);
            } else {
              this.toastService.info('Loading existing feedback form for editing.');
            }
          } else {
            this.sharedFeedback.set(trainingId, false);
            this.feedbackSharedBy.delete(trainingId);
          }
          this.setTrainerZoneView('feedbackForm');
        },
        error: () => {
          this.sharedFeedback.set(trainingId, false);
          this.feedbackSharedBy.delete(trainingId);
          this.setTrainerZoneView('feedbackForm');
        }
      });
    } else {
      this.setTrainerZoneView('feedbackForm');
    }
  }

  resetNewAssignmentForm(): void {
    this.newAssignment = {
      trainingId: null,
      title: '',
      description: '',
      questions: []
    };
  }

  submitAssignment(): void {
    if (!this.newAssignment.trainingId || !this.newAssignment.title.trim() || this.newAssignment.questions.length === 0) {
      this.toastService.warning('Please select a training, provide a title, and add at least one question.', 'Validation Error');
      return;
    }

    for (const q of this.newAssignment.questions) {
      if (!q.text.trim()) {
        this.toastService.warning('Please ensure all questions have text.', 'Validation Error');
        return;
      }
      if (q.type === 'single-choice' || q.type === 'multiple-choice') {
        if (q.options.some(opt => !opt.text.trim())) {
          this.toastService.warning('Please ensure all options have text.', 'Validation Error');
          return;
        }
        if (!q.options.some(opt => opt.isCorrect)) {
          this.toastService.warning(`Please mark at least one correct answer for the question: "${q.text}"`, 'Validation Error');
          return;
        }
      }
    }

    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const payload = {
      training_id: this.newAssignment.trainingId,
      title: this.newAssignment.title,
      description: this.newAssignment.description || '',
      questions: this.newAssignment.questions
    };

    this.http.post(this.apiService.sharedAssignmentsUrl, payload, { headers }).subscribe({
      next: (response: any) => {
        if (this.newAssignment.trainingId) {
          this.sharedAssignments.set(this.newAssignment.trainingId, true);
          // Store who shared it
          if (response.trainer_username) {
            this.assignmentSharedBy.set(this.newAssignment.trainingId, response.trainer_username);
          }
        }
        this.toastService.success('Assignment shared successfully!');
        this.setTrainerZoneView('overview');
        // Refresh the trainings to update the UI
        if (this.activeTab === 'trainerZone') {
          this.fetchScheduledTrainings();
        }
      },
      error: (err) => {
        console.error('Failed to share assignment:', err);
        if (err.status === 403) {
          if (this.newAssignment.trainingId) {
            this.checkSharedStatus(this.newAssignment.trainingId);
            // Use setTimeout to allow checkSharedStatus to complete
            setTimeout(() => {
              if (this.isAssignmentShared(this.newAssignment.trainingId!)) {
                const sharedBy = this.getAssignmentSharedBy(this.newAssignment.trainingId!);
                this.toastService.warning(`Assignment has already been shared for this training by your co-trainer (${sharedBy}). You can update it by modifying the existing assignment.`);
              } else {
                this.toastService.error('You can only share assignments for trainings you have scheduled.');
              }
            }, 500);
          } else {
            this.toastService.error('You can only share assignments for trainings you have scheduled.');
          }
        } else if (err.status === 404) {
          this.toastService.error('Training not found.');
        } else {
          const errorMessage = err.error?.detail || err.message || err.statusText || 'Unknown error';
          this.toastService.error(`Failed to share assignment. Error: ${errorMessage}`);
        }
      }
    });
  }

  addAssignmentQuestion(): void {
    this.newAssignment.questions.push({
      text: '',
      helperText: '',
      type: 'single-choice',
      options: [
        { text: '', isCorrect: false },
        { text: '', isCorrect: false }
      ]
    });
  }

  removeAssignmentQuestion(qIndex: number): void {
    this.newAssignment.questions.splice(qIndex, 1);
  }

  onQuestionTypeChange(question: AssignmentQuestion): void {
    if ((question.type === 'single-choice' || question.type === 'multiple-choice') && question.options.length === 0) {
      question.options.push({ text: '', isCorrect: false }, { text: '', isCorrect: false });
    }
    if (question.type === 'single-choice') {
        let firstCorrectFound = false;
        question.options.forEach(opt => {
            if (opt.isCorrect) {
                if (firstCorrectFound) {
                    opt.isCorrect = false;
                }
                firstCorrectFound = true;
            }
        });
    }
  }

  addOptionToQuestion(qIndex: number): void {
    this.newAssignment.questions[qIndex].options.push({ text: '', isCorrect: false });
  }

  removeOptionFromQuestion(qIndex: number, oIndex: number): void {
    this.newAssignment.questions[qIndex].options.splice(oIndex, 1);
  }

  toggleCorrectOption(qIndex: number, oIndex: number): void {
    const question = this.newAssignment.questions[qIndex];
    if (question.type === 'single-choice') {
      question.options.forEach((opt, index) => {
        opt.isCorrect = (index === oIndex);
      });
    } else if (question.type === 'multiple-choice') {
      question.options[oIndex].isCorrect = !question.options[oIndex].isCorrect;
    }
  }

  resetNewFeedbackForm(): void {
    this.newFeedback = { trainingId: null, customQuestions: [] };
  }

  submitFeedback(): void {
    if (!this.newFeedback.trainingId) {
      this.toastService.warning('Please select a training for the feedback form.', 'Validation Error');
      return;
    }
    const finalCustomQuestions = this.newFeedback.customQuestions
        .filter(q => q.text.trim() !== '')
        .map(q => ({
            ...q,
            options: q.options.filter(opt => opt.trim() !== '')
        }))
        .filter(q => q.options.length > 0);

    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const payload = {
      training_id: this.newFeedback.trainingId,
      defaultQuestions: this.defaultFeedbackQuestions.map(q => ({
        text: q.text,
        options: q.options,
        isDefault: q.isDefault
      })),
      customQuestions: finalCustomQuestions
    };

    this.http.post(this.apiService.sharedFeedbackUrl, payload, { headers }).subscribe({
      next: (response: any) => {
        if (this.newFeedback.trainingId) {
          this.sharedFeedback.set(this.newFeedback.trainingId, true);
          // Store who shared it
          if (response.trainer_username) {
            this.feedbackSharedBy.set(this.newFeedback.trainingId, response.trainer_username);
          }
        }
        this.toastService.success('Feedback form shared successfully!');
        this.setTrainerZoneView('overview');
        // Refresh the trainings to update the UI
        if (this.activeTab === 'trainerZone') {
          this.fetchScheduledTrainings();
        }
      },
      error: (err) => {
        console.error('Failed to share feedback:', err);
        if (err.status === 403) {
          if (this.newFeedback.trainingId) {
            this.checkSharedStatus(this.newFeedback.trainingId!);
            if (this.isFeedbackShared(this.newFeedback.trainingId!)) {
              const sharedBy = this.getFeedbackSharedBy(this.newFeedback.trainingId!);
              this.toastService.warning(`Feedback has already been shared for this training by your co-trainer (${sharedBy}).`);
            } else {
              this.toastService.error('You can only share feedback for trainings you have scheduled.');
            }
          } else {
            this.toastService.error('You can only share feedback for trainings you have scheduled.');
          }
        } else if (err.status === 404) {
          this.toastService.error('Training not found.');
        } else {
          this.toastService.error(`Failed to share feedback. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  addCustomQuestion(): void {
    this.newFeedback.customQuestions.push({
      text: '',
      options: [''],
      isDefault: false
    });
  }

  removeCustomQuestion(index: number): void {
    this.newFeedback.customQuestions.splice(index, 1);
  }

  addOption(questionIndex: number): void {
    this.newFeedback.customQuestions[questionIndex].options.push('');
  }

  removeOption(questionIndex: number, optionIndex: number): void {
    this.newFeedback.customQuestions[questionIndex].options.splice(optionIndex, 1);
  }

  trackByFn(index: any, item: any) {
    return index;
  }

  // --- Missing Methods ---
  viewAssignment(training: any): void {
    // TODO: Implement view assignment functionality
    console.log('Viewing assignment for training:', training);
  }

  giveFeedback(training: any): void {
    // TODO: Implement give feedback functionality
    console.log('Giving feedback for training:', training);
  }

  // --- Calendar logic ---
  generateCalendar(): void {
    const year = this.currentDate.getFullYear();
    const month = this.currentDate.getMonth();
    this.calendarMonth = this.currentDate.toLocaleString('default', { month: 'long' });
    this.calendarYear = year;

    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startDay = firstDay.getDay();

    this.calendarDays = [];
    for (let i = 0; i < startDay; i++) {
      this.calendarDays.push(null);
    }
    for (let i = 1; i <= daysInMonth; i++) {
      this.calendarDays.push(new Date(year, month, i));
    }
  }

  previousMonth(): void {
    this.currentDate.setMonth(this.currentDate.getMonth() - 1);
    this.generateCalendar();
  }

  nextMonth(): void {
    this.currentDate.setMonth(this.currentDate.getMonth() + 1);
    this.generateCalendar();
  }

  isToday(date: Date | null): boolean {
    if (!date) return false;
    const today = new Date();
    return date.getDate() === today.getDate() &&
      date.getMonth() === today.getMonth() &&
      date.getFullYear() === today.getFullYear();
  }

  getEventForDate(date: Date | null, events: CalendarEvent[]): CalendarEvent[] {
    if (!date) return [];
    return events.filter(event =>
      event.date.getDate() === date.getDate() &&
      event.date.getMonth() === date.getMonth() &&
      event.date.getFullYear() === date.getFullYear()
    );
  }

  trackByEventId(index: number, event: CalendarEvent): string {
    return event.title + event.trainer + event.date.getTime();
  }


  // --- Training Enrollment ---
  enrollInTraining(training: TrainingDetail): void {
    // This is a placeholder method. In a real app, you would make an API call here.
    this.toastService.success(`Enrolled in "${training.training_name}" successfully!`, 'Enrollment Successful');
  }

  // --- File Management Methods ---
  uploadQuestionFile(trainingId: number, event: any): void {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      this.toastService.error('Only PDF files are allowed');
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('training_id', trainingId.toString());

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });

    this.http.post(this.apiService.uploadQuestionFileUrl, formData, { headers }).subscribe({
      next: (response: any) => {
        this.toastService.success('Question file uploaded successfully');
        this.questionFilesUploaded.set(trainingId, true);
        event.target.value = ''; // Reset file input
      },
      error: (err) => {
        console.error('Failed to upload question file:', err);
        this.toastService.error(err.error?.detail || 'Failed to upload question file');
      }
    });
  }

  hasQuestionFile(trainingId: number): boolean {
    return this.questionFilesUploaded.get(trainingId) || false;
  }

  viewSolutions(trainingId: number): void {
    this.selectedTrainingForSolutions = trainingId;
    this.showSolutionsModal = true;
    this.loadSolutions(trainingId);
  }

  loadSolutions(trainingId: number): void {
    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      return;
    }

    this.isLoadingSolutions = true;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.http.get<any[]>(this.apiService.trainerSolutionsUrl(trainingId), { headers }).subscribe({
      next: (solutions) => {
        this.solutionsList = solutions || [];
        this.trainerSolutions.set(trainingId, solutions || []);
        this.isLoadingSolutions = false;
        if (solutions.length === 0) {
          this.toastService.info('No solutions submitted yet for this training');
        }
      },
      error: (err) => {
        console.error('Failed to fetch solutions:', err);
        this.solutionsList = [];
        this.trainerSolutions.set(trainingId, []);
        this.isLoadingSolutions = false;
        this.toastService.error(err.error?.detail || 'Failed to load solutions');
      }
    });
  }

  closeSolutionsModal(): void {
    this.showSolutionsModal = false;
    this.selectedTrainingForSolutions = null;
    this.solutionsList = [];
  }

  downloadSolutionFile(trainingId: number, employeeId: string, fileName: string): void {
    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get(this.apiService.solutionFileUrl(trainingId, employeeId), { 
      headers, 
      responseType: 'blob' 
    }).subscribe({
      next: (blob: Blob) => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = fileName || `solution_${trainingId}_${employeeId}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        this.toastService.success('Solution file downloaded successfully');
      },
      error: (err) => {
        console.error('Failed to download solution file:', err);
        this.toastService.error(err.error?.detail || 'Failed to download solution file');
      }
    });
  }

  getTrainingName(trainingId: number): string {
    const training = this.myTrainings.find(t => t.id === trainingId);
    return training?.training_name || 'Training';
  }
}