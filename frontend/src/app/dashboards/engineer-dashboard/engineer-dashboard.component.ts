/**
 * Engineer Dashboard Component
 * 
 * Purpose: Main dashboard for employees/engineers to manage their skills, trainings, and assignments
 * 
 * Features:
 * - Skill Management:
 *   - View core skills and competency levels (L1-L5)
 *   - Track skill gaps and progress
 *   - Add and manage additional self-reported skills
 *   - View detailed skill breakdowns by competency area
 * 
 * - Training Management:
 *   - Browse training catalog with filtering (skill, level, date)
 *   - View assigned trainings
 *   - Request training approvals from managers
 *   - View training calendar (list and calendar views)
 *   - Track upcoming trainings
 * 
 * - Assignment & Assessment:
 *   - Take training assignments/exams
 *   - View assignment results and scores
 *   - Submit feedback for completed trainings
 *   - View manager performance feedback
 * 
 * - Trainer Zone (if user is a trainer):
 *   - Schedule new trainings
 *   - Create and share assignments
 *   - Create and share feedback forms
 *   - View shared content
 * 
 * - Dashboard Metrics:
 *   - Skill progress percentage
 *   - Skills met vs. skills gap count
 *   - Upcoming trainings count
 *   - Next training information
 * 
 * Key Sections:
 * 1. Dashboard Tab: Overview with metrics and upcoming trainings
 * 2. My Skills Tab: Core and additional skills management
 * 3. Training Catalog Tab: Browse and filter all available trainings
 * 4. Assigned Trainings Tab: View trainings assigned to the employee
 * 5. Training Requests Tab: Manage training approval requests
 * 6. Trainer Zone Tab: Trainer-specific functionality (if applicable)
 * 
 * State Management:
 * - Uses Maps to track assignment/feedback submission status
 * - Caches training data for performance
 * - Manages multiple filter states for different views
 * 
 * API Integration:
 * - Uses ApiService for centralized endpoint management
 * - Implements proper error handling with ToastService
 * - Handles authentication via AuthService
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { ApiService } from '../../services/api.service';
import { map, catchError } from 'rxjs/operators';
import { of } from 'rxjs';
import { trigger, style, animate, transition, query, stagger } from '@angular/animations';
import { ToastService, ToastMessage } from '../../services/toast.service';
import { NotificationService } from '../../services/notification.service';
import { Skill, ModalSkill } from '../../models/skill.model';
import { TrainingDetail, TrainingRequest, CalendarEvent } from '../../models/training.model';
import { Assignment, AssignmentQuestion, QuestionOption, UserAnswer, QuestionResult, AssignmentResult, FeedbackQuestion } from '../../models/assignment.model';

/**
 * Type definitions for skill level structure
 */
type LevelBlock = { level: number; items: string[] };
type Section = { title: string; subtitle?: string; levels: LevelBlock[] };

@Component({
  selector: 'app-engineer-dashboard',
  templateUrl: './engineer-dashboard.component.html',
  styleUrls: ['./engineer-dashboard.component.css'],
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
export class EngineerDashboardComponent implements OnInit {
  // --- Component State & Filters ---
  skillSearch: string = '';
  skillStatusFilter: string = '';
  skillNameFilter: string = '';
  skillNames: string[] = [];
  mySkillsView: 'core' | 'additional' = 'core'; // Toggle between core and additional skills
  userName: string = '';
  employeeId: string = '';
  employeeName: string = '';
  skills: Skill[] = [];
  skillGaps: Skill[] = [];
  totalSkills: number = 0;
  skillsMet: number = 0;
  skillsGap: number = 0;
  progressPercentage: number = 0;
  isLoading: boolean = true;
  errorMessage: string = '';
  activeTab: string = 'dashboard';

  // --- Skills Modal State ---
  showSkillsModal: boolean = false;
  modalTitle: string = '';
  modalSkills: ModalSkill[] = [];

  // --- Additional (Self-Reported) Skills ---
  additionalSkills: any[] = [];
  newSkill = {
    name: '',
    level: 'Beginner',
    category: 'Technical',
    description: ''
  };
  showAddSkillForm: boolean = false;
  editingSkillId: number | null = null;
  skillLevels = ['Beginner', 'Intermediate', 'Advanced', 'Expert'];
  skillCategories = ['Technical', 'Soft Skills', 'Leadership', 'Communication', 'Project Management', 'Other'];
  skillCategoryLevels: string[] = ['L1', 'L2', 'L3', 'L4', 'L5'];

  // --- Levels Definitions ---
  levelsSearch = '';
  selectedSkill = '';
  public expandedLevels = new Set<string>();
  public expandedSkill: string | null = null; // <<< NEW PROPERTY FOR ACCORDION
  levelHeaders = [
    { num: 1, title: 'Beginner' },
    { num: 2, title: 'Basic' },
    { num: 3, title: 'Intermediate' },
    { num: 4, title: 'Advanced' },
    { num: 5, title: 'Expert' }
  ];


  // --- Trainings & Catalog ---
  allTrainings: TrainingDetail[] = [];
  myTrainingsFromBackend: TrainingDetail[] = []; // Trainings where user is trainer (from backend)
  private _myTrainingsCache: TrainingDetail[] = [];
  private _myTrainingsCacheKey: string = '';
  assignedTrainings: TrainingDetail[] = [];
  dashboardUpcomingTrainings: TrainingDetail[] = [];
  trainingRequests: TrainingRequest[] = [];
  assignmentSubmissionStatus: Map<number, boolean> = new Map(); // Track which trainings have submitted assignments
  assignmentScores: Map<number, number> = new Map(); // Track scores for each training
  feedbackSubmissionStatus: Map<number, boolean> = new Map(); // Track which trainings have submitted feedback
  
  // Manager Performance Feedback
  managerFeedback: any[] = [];
  isLoadingFeedback: boolean = false;
  showSkillFeedbackModal: boolean = false;
  selectedSkillForFeedback: string = '';
  selectedFeedbackSkillType: string = ''; // Currently selected skill type tab (L1-L5) in feedback modal
  skillFeedbackList: any[] = [];
  skillFeedbackByType: Map<string, any[]> = new Map(); // Grouped by skill type (L1-L5)
  showFeedbackHistoryModal: boolean = false;
  selectedSkillTypeForHistory: string = '';
  selectedSkillNameForHistory: string = '';
  feedbackHistoryList: any[] = [];
  trainingSearch: string = '';
  trainingSkillFilter: string = 'All';
  trainingLevelFilter: string = 'All';
  trainingDateFilter: string = '';
  trainingCatalogView: 'list' | 'calendar' = 'list';

  // --- Assigned Trainings Filters ---
  assignedSearch: string = '';
  assignedSkillFilter: string = 'All';
  assignedLevelFilter: string = 'All';
  assignedDateFilter: string = '';
  assignedTrainingsView: 'list' | 'calendar' = 'list';

  // --- Training Requests Filters ---
  requestStatusFilter: string = 'all';
  requestSearch: string = '';

  // --- Calendar & Dashboard Metrics ---
  allTrainingsCalendarEvents: CalendarEvent[] = [];
  assignedTrainingsCalendarEvents: CalendarEvent[] = [];
  badges: Skill[] = [];
  upcomingTrainingsCount: number = 0;
  nextTrainingTitle: string = '';
  currentDate: Date = new Date();
  calendarDays: (Date | null)[] = [];
  calendarMonth: string = '';
  calendarYear: number = 2025;

  // --- Trainer Zone ---
  isTrainer: boolean = false;
  trainerZoneView: 'overview' | 'assignmentForm' | 'feedbackForm' = 'overview';
  showScheduleTrainingModal: boolean = false;
  sharedAssignments: Map<number, boolean> = new Map(); // Track which trainings have assignments shared
  sharedFeedback: Map<number, boolean> = new Map(); // Track which trainings have feedback shared
  assignmentSharedBy: Map<number, string> = new Map(); // Track who shared the assignment
  feedbackSharedBy: Map<number, string> = new Map(); // Track who shared the feedback
  trainingCandidates: Map<number, {employee_empid: string, employee_name: string, attended: boolean}[]> = new Map(); // Store candidates for each training
  showAttendanceModal: boolean = false;
  selectedTrainingForAttendance: number | null = null;
  attendanceCandidates: {employee_empid: string, employee_name: string, attended: boolean}[] = [];
  isMarkingAttendance: boolean = false;
  
  // Attendance success popup
  showAttendanceSuccessPopup: boolean = false;
  attendanceSuccessData: {
    trainingName: string;
    attendedCount: number;
    absentCount: number;
    totalCount: number;
    attendedNames: string;
  } | null = null;
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

  // --- Static Data ---
  sections: Section[] = [
    {
      title: 'EXAM',
      levels: [
        { level: 1, items: ['Launch EXAM', 'Test execution', 'Exporting reports'] },
        { level: 2, items: ['Implement test cases', 'Create collections', 'EXAM configuration', 'DOORS synchronization'] },
        { level: 3, items: ['Create short names', 'NeKeDa reporting', 'Debugging in EXAM'] },
        { level: 4, items: ['Implement libraries & common sequences', 'Know-how on libraries (1–4)', 'Create baselines', 'Release configuration', 'Update variable mapping', 'System configurations'] },
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

  private readonly API_ENDPOINT = '/data/engineer';

  constructor(
    private http: HttpClient,
    private router: Router,
    private authService: AuthService,
    private apiService: ApiService,
    private cdr: ChangeDetectorRef,
    private toastService: ToastService,
    private notificationService: NotificationService
  ) {}

  ngOnInit(): void {
    // Check authentication and fetch all data
    const token = this.authService.getToken();
    if (!token) {
      console.warn('No token found on component init - redirecting to login');
      this.router.navigate(['/login']);
      return;
    }
    // Always fetch all data fresh on component initialization
    // This ensures data consistency after logout/login cycles
    this.fetchDashboardData();
    this.fetchScheduledTrainings();
    this.fetchMyTrainings(); // Fetch trainings where user is trainer
    this.fetchAssignedTrainings();
    this.fetchTrainingRequests();
    // Load additional skills
    this.loadAdditionalSkills();
    // Load manager feedback
    this.fetchManagerFeedback();
    // Initialize notifications
    this.notificationService.initialize();
  }

  // Fetch manager performance feedback
  fetchManagerFeedback(): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.isLoadingFeedback = true;
    
    this.http.get<any[]>(this.apiService.getUrl('/shared-content/employee/performance-feedback'), { headers }).subscribe({
      next: (feedback) => {
        // Sort feedback by updated_at (most recent first)
        this.managerFeedback = (feedback || []).sort((a, b) => {
          const dateA = new Date(a.updated_at || a.created_at || 0).getTime();
          const dateB = new Date(b.updated_at || b.created_at || 0).getTime();
          return dateB - dateA; // Descending order (newest first)
        });
        // Debug: Log all feedback to verify backend is returning all entries with all fields
        console.log('All manager feedback received:', this.managerFeedback);
        console.log('Total feedback entries from backend:', this.managerFeedback.length);
        // Log detailed field information for each feedback entry
        this.managerFeedback.forEach((fb, index) => {
          console.log(`Feedback ${index + 1}:`, {
            id: fb.id,
            training_id: fb.training_id,
            training_name: fb.training_name,
            improvement_areas: fb.improvement_areas,
            strengths: fb.strengths,
            additional_comments: fb.additional_comments,
            overall_performance: fb.overall_performance,
            has_improvement_areas: !!fb.improvement_areas,
            has_strengths: !!fb.strengths,
            has_additional_comments: !!fb.additional_comments
          });
        });
        this.isLoadingFeedback = false;
      },
      error: (err) => {
        console.error('Failed to fetch manager feedback:', err);
        this.managerFeedback = [];
        this.isLoadingFeedback = false;
      }
    });
  }

  // Get rating label
  getRatingLabel(rating: number | null | undefined): string {
    if (!rating) return 'Not Rated';
    const labels: { [key: number]: string } = {
      1: 'Poor',
      2: 'Below Average',
      3: 'Average',
      4: 'Good',
      5: 'Excellent'
    };
    return `${rating} - ${labels[rating] || 'Unknown'}`;
  }

  // Get rating color class
  getRatingColorClass(rating: number | null | undefined): string {
    if (!rating) return 'bg-slate-100 text-slate-600';
    if (rating >= 4) return 'bg-green-100 text-green-700';
    if (rating >= 3) return 'bg-yellow-100 text-yellow-700';
    return 'bg-red-100 text-red-700';
  }

  // Check if feedback was recently updated (within last 24 hours)
  isFeedbackRecentlyUpdated(feedback: any): boolean {
    if (!feedback.updated_at) return false;
    const updatedDate = new Date(feedback.updated_at);
    const now = new Date();
    const hoursDiff = (now.getTime() - updatedDate.getTime()) / (1000 * 60 * 60);
    // Show "Updated" badge if feedback was updated within last 24 hours
    return hoursDiff < 24 && feedback.updated_at !== feedback.created_at;
  }

  // Get feedback for a specific skill by matching training skill names
  getFeedbackForSkill(skillName: string): any[] {
    // Handle edge cases: empty skill name, no feedback data, or no trainings
    if (!skillName || !skillName.trim()) {
      return [];
    }

    // If no feedback has been loaded yet, return empty array
    if (!this.managerFeedback || this.managerFeedback.length === 0) {
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

    // Debug: Log matching information
    console.log(`Matching trainings for skill "${skillName}":`, matchingTrainings.map(t => ({ id: t.id, name: t.training_name, skill: t.skill })));

    if (matchingTrainings.length === 0) {
      console.log(`No trainings found matching skill "${skillName}"`);
      return [];
    }

    // Get training IDs for matching trainings
    const trainingIds = matchingTrainings.map(t => t.id).filter(id => id != null);

    if (trainingIds.length === 0) {
      return [];
    }

    // Create a map of training_id to training for quick lookup
    const trainingMap = new Map<number, any>();
    matchingTrainings.forEach(t => {
      if (t.id != null) {
        trainingMap.set(t.id, t);
      }
    });

    // Filter feedback where training_id matches and enrich with training info
    const filteredFeedback = this.managerFeedback
      .filter(feedback => {
        if (!feedback || !feedback.training_id) return false;
        return trainingIds.includes(feedback.training_id);
      })
      .map(feedback => {
        const training = trainingMap.get(feedback.training_id);
        return {
          ...feedback,
          skill_category: training?.skill_category || null,
          training_name: training?.training_name || feedback.training_name
        };
      });

    // Debug: Log filtered feedback information
    console.log(`Filtered feedback for skill "${skillName}" (training IDs: ${trainingIds.join(', ')}):`, filteredFeedback.length, 'entries');
    filteredFeedback.forEach((fb, idx) => {
      console.log(`  Feedback ${idx + 1}:`, {
        training_id: fb.training_id,
        training_name: fb.training_name,
        skill_category: fb.skill_category,
        has_improvement_areas: !!fb.improvement_areas,
        has_strengths: !!fb.strengths,
        has_additional_comments: !!fb.additional_comments
      });
    });

    // Group feedback by training_id and add update numbers
    const feedbackByTraining = new Map<number, any[]>();
    filteredFeedback.forEach(feedback => {
      const trainingId = feedback.training_id;
      if (!feedbackByTraining.has(trainingId)) {
        feedbackByTraining.set(trainingId, []);
      }
      feedbackByTraining.get(trainingId)!.push(feedback);
    });

    // Add update numbers to each feedback entry (1 = most recent, 2 = second most recent, etc.)
    const enrichedFeedback: any[] = [];
    feedbackByTraining.forEach((feedbackList, trainingId) => {
      // Sort by updated_at descending (most recent first)
      feedbackList.sort((a, b) => {
        const dateA = new Date(a.updated_at || a.created_at || 0).getTime();
        const dateB = new Date(b.updated_at || b.created_at || 0).getTime();
        return dateB - dateA;
      });

      // Add update number and total updates count
      feedbackList.forEach((feedback, index) => {
        // Ensure all feedback fields are preserved when enriching
        const enrichedEntry = {
          ...feedback,
          updateNumber: index + 1,
          totalUpdates: feedbackList.length
        };
        // Debug: Log to verify all fields are preserved
        if (index === 0) {
          console.log('Enriched feedback entry (most recent):', {
            id: enrichedEntry.id,
            training_id: enrichedEntry.training_id,
            training_name: enrichedEntry.training_name,
            improvement_areas: enrichedEntry.improvement_areas,
            strengths: enrichedEntry.strengths,
            additional_comments: enrichedEntry.additional_comments,
            improvement_areas_length: enrichedEntry.improvement_areas?.length || 0,
            strengths_length: enrichedEntry.strengths?.length || 0,
            additional_comments_length: enrichedEntry.additional_comments?.length || 0
          });
        }
        enrichedFeedback.push(enrichedEntry);
      });
    });

    // Sort all feedback to keep same training together, with most recent training first
    // Within each training, feedback is already sorted by update number (1 = most recent)
    enrichedFeedback.sort((a, b) => {
      // First, sort by training_id to keep same training together
      if (a.training_id !== b.training_id) {
        // Get the most recent date for each training to determine which training comes first
        const dateA = new Date(a.updated_at || a.created_at || 0).getTime();
        const dateB = new Date(b.updated_at || b.created_at || 0).getTime();
        // If same training, maintain order; if different, most recent training first
        return dateB - dateA;
      }
      // Within same training, sort by update number (1 = most recent)
      return a.updateNumber - b.updateNumber;
    });

    return enrichedFeedback;
  }

  // Check if a skill has feedback available
  hasFeedbackForSkill(skillName: string): boolean {
    return this.getFeedbackForSkill(skillName).length > 0;
  }

  // Group feedback by skill type (L1-L5)
  groupFeedbackBySkillType(feedbackList: any[]): Map<string, any[]> {
    const grouped = new Map<string, any[]>();
    
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
  getLatestFeedbackForType(feedbackList: any[]): any | null {
    if (!feedbackList || feedbackList.length === 0) return null;
    
    // List is already sorted by updated_at descending
    return feedbackList[0];
  }

  // Get skill type entries as array for template iteration
  getSkillTypeEntries(): Array<{key: string, value: any[]}> {
    return Array.from(this.skillFeedbackByType.entries()).map(([key, value]) => ({
      key,
      value
    }));
  }

  // Get filtered skill type entries based on selected tab
  getFilteredSkillTypeEntries(): Array<{key: string, value: any[]}> {
    const entries = this.getSkillTypeEntries();
    if (!this.selectedFeedbackSkillType) {
      return entries;
    }
    return entries.filter(entry => entry.key === this.selectedFeedbackSkillType);
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

  // Open feedback modal for a specific skill
  openSkillFeedbackModal(skillName: string): void {
    if (!skillName) return;
    this.selectedSkillForFeedback = skillName;
    this.skillFeedbackList = this.getFeedbackForSkill(skillName);
    
    // Group feedback by skill type
    this.skillFeedbackByType = this.groupFeedbackBySkillType(this.skillFeedbackList);
    const typeKeys = Array.from(this.skillFeedbackByType.keys());
    // Default selected tab to first available type (if any)
    this.selectedFeedbackSkillType = typeKeys.length > 0 ? typeKeys[0] : '';
    
    // Debug: Log all feedback entries to verify all are being returned with complete data
    console.log(`Feedback for skill "${skillName}":`, this.skillFeedbackList);
    console.log(`Total feedback entries: ${this.skillFeedbackList.length}`);
    console.log(`Grouped by type:`, Array.from(this.skillFeedbackByType.entries()).map(([type, list]) => ({
      type,
      count: list.length
    })));
    
    this.showSkillFeedbackModal = true;
  }

  // Open history modal for a specific skill type
  openFeedbackHistoryModal(skillName: string, skillType: string): void {
    if (!skillName || !skillType) return;
    this.selectedSkillNameForHistory = skillName;
    this.selectedSkillTypeForHistory = skillType;
    
    // Get all feedback for this skill type
    const allFeedback = this.getFeedbackForSkill(skillName);
    this.feedbackHistoryList = allFeedback.filter(fb => 
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
    this.selectedSkillNameForHistory = '';
    this.feedbackHistoryList = [];
  }

  // Close feedback modal
  closeSkillFeedbackModal(): void {
    this.showSkillFeedbackModal = false;
    this.selectedSkillForFeedback = '';
    this.skillFeedbackList = [];
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

  getEventForDate(date: Date | null, events: CalendarEvent[]): CalendarEvent | undefined {
    if (!date) return undefined;
    return events.find(event =>
      event.date.getDate() === date.getDate() &&
      event.date.getMonth() === date.getMonth() &&
      event.date.getFullYear() === date.getFullYear()
    );
  }

  // --- Data Fetching & Processing ---
  fetchDashboardData(): void {
    this.isLoading = true;
    this.errorMessage = '';

    const token = this.authService.getToken();
    const userRole = this.authService.getRole();

    if (!token) {
      // Silently redirect to login on token expiration
      this.errorMessage = '';
      this.isLoading = false;
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    if (userRole !== 'employee') {
      this.errorMessage = `Invalid role: ${userRole}. Expected 'employee'.`;
      this.isLoading = false;
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });

    this.http.get<any>(this.apiService.getUrl(this.API_ENDPOINT), { headers }).pipe(
      map(response => {
        this.employeeId = response.username;
        this.employeeName = response.employee_name || 'Employee';
        this.userName = this.employeeName || this.employeeId;
        this.skills = response.skills;
        this.isTrainer = response.employee_is_trainer || false;
        if (this.skills && this.skills.length > 0) {
          this.skillNames = Array.from(new Set(this.skills.map(skill => skill.skill))).sort();
        }
        this.processDashboardData();
        this.loadAdditionalSkills();
        this.isLoading = false;
        return response;
      }),
      catchError(err => {
        if (err.status === 401) {
          // Silently handle token expiration - redirect to login without showing error
          this.errorMessage = '';
          this.authService.logout();
          this.router.navigate(['/login']);
        } else {
          // Mock data for display purposes if API fails
          this.employeeName = 'Employee';
          this.userName = 'Employee';
          this.employeeId = 'employee';
          this.processDashboardData(); // Will use default values
          this.dashboardUpcomingTrainings = [{ id: 1, training_name: 'Sample Training', training_date: '2025-10-15', time: '10:00 AM', trainer_name: 'Manager', training_type: 'Online' }];
          this.errorMessage = `Failed to load live data. Displaying sample data.`;
        }
        this.isLoading = false;
        return of(null);
      })
    ).subscribe();
  }

  processDashboardData(): void {
    // Core Skills count is based on static sections (10 skills)
    this.totalSkills = this.sections.length;
    
    if (!this.skills || this.skills.length === 0) {
      // Use hardcoded defaults if API fails or returns no skills
      this.skillsMet = 3; // Default value for Skills Met
      this.skillsGap = 7; // Default value for Skills Gap
      this.progressPercentage = 30; // 3/10 = 30%
      this.skillGaps = [];
      this.badges = [];
      return;
    }

    // Skills Met count is based on API skills with status "Met"
    this.skillsMet = this.skills.filter(s => s.status === 'Met').length;
    this.skillsGap = this.skills.filter(s => s.status === 'Gap').length;
    this.progressPercentage = this.totalSkills > 0
      ? Math.round((this.skillsMet / this.totalSkills) * 100)
      : 0;

    this.skillGaps = this.skills.filter(s => s.status === 'Gap');
    this.badges = this.skills.filter(s => s.status === 'Met');
  }

  processDashboardTrainings(): void {
    if (!this.assignedTrainings || this.assignedTrainings.length === 0) {
        this.dashboardUpcomingTrainings = [];
        return;
    }
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    this.dashboardUpcomingTrainings = this.assignedTrainings
        .filter(t => t.training_date && new Date(t.training_date) >= today)
        .sort((a, b) => {
            return new Date(a.training_date!).getTime() - new Date(b.training_date!).getTime();
        });
  }

  // --- Training Data & Filtering ---
  fetchScheduledTrainings(): void {
    const token = this.authService.getToken();
    if (!token) {
      console.warn('No token available for fetching scheduled trainings');
      // Don't clear data if token is missing - maintain consistency
      return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get<TrainingDetail[]>(this.apiService.getUrl('/trainings/'), { headers }).subscribe({
      next: (response) => {
        this.allTrainings = response || [];
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
        // Refresh my trainings from backend when all trainings are updated
        this.fetchMyTrainings();
      },
      error: (err) => {
        console.error('Failed to fetch scheduled trainings:', err);
        console.error('Error details:', {
          status: err.status,
          statusText: err.statusText,
          error: err.error
        });
        // Don't clear existing data on error - maintain consistency
        // Only clear if unauthorized - silently redirect without showing error
        if (err.status === 401 || err.status === 403) {
          this.authService.logout();
          this.router.navigate(['/login']);
        }
      }
    });
  }

  fetchMyTrainings(): void {
    const token = this.authService.getToken();
    if (!token) {
      console.warn('No token available for fetching my trainings');
      return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get<TrainingDetail[]>(this.apiService.myTrainingsUrl, { headers }).subscribe({
      next: (response) => {
        this.myTrainingsFromBackend = response || [];
        // Clear cache when my trainings are updated
        this._myTrainingsCache = [];
        this._myTrainingsCacheKey = '';
        // Check shared status and fetch candidates for all trainings
        this.myTrainingsFromBackend.forEach(training => {
          this.checkSharedStatus(training.id);
          this.fetchTrainingCandidates(training.id);
        });
      },
      error: (err) => {
        console.error('Failed to fetch my trainings:', err);
        console.error('Error details:', {
          status: err.status,
          statusText: err.statusText,
          error: err.error
        });
        // If unauthorized, silently redirect to login
        if (err.status === 401 || err.status === 403) {
          this.authService.logout();
          this.router.navigate(['/login']);
        } else {
          // For other errors, keep existing data
          console.warn('Non-auth error - keeping existing my trainings data');
        }
      }
    });
  }

  fetchAssignedTrainings(): void {
    const token = this.authService.getToken();
    if (!token) {
      console.warn('No token available for fetching assigned trainings');
      // Don't clear data if token is missing - maintain consistency
      // Only clear if explicitly unauthorized
      return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get<TrainingDetail[]>(this.apiService.getUrl('/assignments/my'), { headers }).subscribe({
      next: (response) => {
        this.assignedTrainings = (response || []).map(t => ({ ...t, assignmentType: 'personal' as const }));
        this.assignedTrainingsCalendarEvents = this.assignedTrainings
            .filter(t => t.training_date)
            .map(t => ({
                date: new Date(t.training_date as string),
                title: t.training_name,
                trainer: t.trainer_name || 'N/A'
            }));
        this.generateCalendar();
        this.processDashboardTrainings();
        // Check submission status for all assigned trainings
        this.checkSubmissionStatuses();
      },
      error: (err) => {
        console.error('Failed to fetch assigned trainings:', err);
        console.error('Error details:', {
          status: err.status,
          statusText: err.statusText,
          error: err.error
        });
        // If unauthorized, silently redirect to login (don't clear data - let component reinitialize on next login)
        if (err.status === 401 || err.status === 403) {
          // Don't clear data arrays - they will be refreshed when user logs back in
          this.authService.logout();
          this.router.navigate(['/login']);
        } else {
          // For other errors, don't clear data - maintain consistency
          // Data will remain from previous successful fetch
          console.warn('Non-auth error - keeping existing assigned trainings data');
        }
      }
    });
  }

  checkSubmissionStatuses(): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    // Check assignment submission status for each training
    this.assignedTrainings.forEach(training => {
      // Check assignment submission status
      // Only log errors if they're not 403 (403 means attendance not marked, which is expected)
      this.http.get(this.apiService.getUrl(`/shared-content/assignments/${training.id}/result`), { headers }).subscribe({
        next: (result: any) => {
          if (result) {
            this.assignmentSubmissionStatus.set(training.id, true);
            this.assignmentScores.set(training.id, result.score || 0);
          }
        },
        error: (err) => {
          // 403 means attendance not marked yet - this is expected, don't log as error
          if (err.status === 403) {
            // Attendance not marked - silently set as not submitted
            this.assignmentSubmissionStatus.set(training.id, false);
            this.assignmentScores.set(training.id, 0);
          } else if (err.status === 404) {
            // No result found - not submitted yet
            this.assignmentSubmissionStatus.set(training.id, false);
            this.assignmentScores.set(training.id, 0);
          } else {
            // Other errors - log only if not 403
            console.warn(`Failed to check assignment status for training ${training.id}:`, err.status);
            this.assignmentSubmissionStatus.set(training.id, false);
            this.assignmentScores.set(training.id, 0);
          }
        }
      });

      // Check feedback submission status
      // Only log errors if they're not 403 (403 means attendance not marked, which is expected)
      this.http.get(this.apiService.getUrl(`/shared-content/feedback/${training.id}/result`), { headers }).subscribe({
        next: (result: any) => {
          if (result) {
            this.feedbackSubmissionStatus.set(training.id, true);
          }
        },
        error: (err) => {
          // 403 means attendance not marked yet - this is expected, don't log as error
          if (err.status === 403) {
            // Attendance not marked - silently set as not submitted
            this.feedbackSubmissionStatus.set(training.id, false);
          } else if (err.status === 404) {
            // No result found - not submitted yet
            this.feedbackSubmissionStatus.set(training.id, false);
          } else {
            // Other errors - log only if not 403
            console.warn(`Failed to check feedback status for training ${training.id}:`, err.status);
            this.feedbackSubmissionStatus.set(training.id, false);
          }
        }
      });
    });
  }

  isAssignmentSubmitted(trainingId: number): boolean {
    return this.assignmentSubmissionStatus.get(trainingId) || false;
  }

  isAssignmentCompleted(trainingId: number): boolean {
    // Assignment is completed only if score is 100%
    const score = this.assignmentScores.get(trainingId) || 0;
    return score === 100;
  }

  getAssignmentScore(trainingId: number): number {
    return this.assignmentScores.get(trainingId) || 0;
  }

  isFeedbackSubmitted(trainingId: number): boolean {
    return this.feedbackSubmissionStatus.get(trainingId) || false;
  }

  isAttendanceMarked(trainingId: number): boolean {
    // Check if attendance has been marked for this training
    const training = this.assignedTrainings.find(t => t.id === trainingId);
    return training?.attendance_marked === true;
  }

  isAttendanceAttended(trainingId: number): boolean {
    // Check if the employee attended the training (attendance marked AND attended = true)
    const training = this.assignedTrainings.find(t => t.id === trainingId);
    return training?.attendance_marked === true && training?.attendance_attended === true;
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
    this.http.get(this.apiService.getUrl(`/shared-content/trainer/assignments/${trainingId}`), { headers }).subscribe({
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
    this.http.get(this.apiService.getUrl(`/shared-content/trainer/feedback/${trainingId}`), { headers }).subscribe({
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

  getAssignmentSharedBy(trainingId: number): string {
    return this.assignmentSharedBy.get(trainingId) || '';
  }

  fetchTrainingCandidates(trainingId: number): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.http.get<{employee_empid: string, employee_name: string, attended: boolean}[]>(
      this.apiService.getTrainingCandidatesUrl(trainingId), 
      { headers }
    ).subscribe({
      next: (candidates) => {
        this.trainingCandidates.set(trainingId, candidates);
        // If modal is open for this training, update the attendance candidates
        if (this.showAttendanceModal && this.selectedTrainingForAttendance === trainingId) {
          this.attendanceCandidates = candidates.map(c => ({ ...c }));
        }
      },
      error: (err) => {
        // If 401, token expired - redirect to login
        if (err.status === 401) {
          this.authService.logout();
          this.router.navigate(['/login']);
        } else if (err.status === 403) {
          // If 403, user is not the trainer - log for debugging
          console.warn(`Access denied for training ${trainingId}. User may not be the trainer.`, err.error?.detail || '');
        } else {
          console.error('Failed to fetch training candidates:', err);
        }
        this.trainingCandidates.set(trainingId, []);
        // If modal is open, clear candidates
        if (this.showAttendanceModal && this.selectedTrainingForAttendance === trainingId) {
          this.attendanceCandidates = [];
        }
      }
    });
  }

  getTrainingCandidates(trainingId: number): {employee_empid: string, employee_name: string, attended: boolean}[] {
    return this.trainingCandidates.get(trainingId) || [];
  }

  openAttendanceModal(trainingId: number): void {
    this.selectedTrainingForAttendance = trainingId;
    const candidates = this.getTrainingCandidates(trainingId);
    
    // If no candidates cached, fetch them now before opening modal
    if (candidates.length === 0) {
      const token = this.authService.getToken();
      if (!token) {
        this.toastService.error('Authentication token missing. Please login again.');
        return;
      }
      const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
      
      // Fetch candidates first, then open modal
      this.http.get<{employee_empid: string, employee_name: string, attended: boolean}[]>(
        this.apiService.getTrainingCandidatesUrl(trainingId), 
        { headers }
      ).subscribe({
        next: (fetchedCandidates) => {
          this.trainingCandidates.set(trainingId, fetchedCandidates);
          this.attendanceCandidates = fetchedCandidates.map(c => ({ ...c }));
          this.showAttendanceModal = true;
        },
        error: (err) => {
          console.error('Failed to fetch training candidates:', err);
          if (err.status === 401) {
            this.authService.logout();
            this.router.navigate(['/login']);
          } else if (err.status === 403) {
            this.toastService.error('You are not authorized to mark attendance for this training.');
          } else {
            this.toastService.error('Failed to load candidates. Please try again.');
          }
          this.attendanceCandidates = [];
          // Still open modal but with empty candidates
          this.showAttendanceModal = true;
        }
      });
    } else {
      // Create a copy for editing
      this.attendanceCandidates = candidates.map(c => ({ ...c }));
      this.showAttendanceModal = true;
    }
  }

  closeAttendanceModal(): void {
    this.showAttendanceModal = false;
    this.selectedTrainingForAttendance = null;
    this.attendanceCandidates = [];
  }
  
  closeAttendanceSuccessPopup(): void {
    this.showAttendanceSuccessPopup = false;
    this.attendanceSuccessData = null;
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
      this.toastService.error('Authentication token missing. Please login again.');
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
          this.toastService.error('Only the trainer of this training can mark attendance.');
        } else if (err.status === 400) {
          this.toastService.error(err.error?.detail || 'Invalid request. Please check the selected candidates.');
        } else {
          this.toastService.error('Failed to mark attendance. Please try again.');
        }
      },
      complete: () => {
        this.isMarkingAttendance = false;
      }
    });
  }

  getFeedbackSharedBy(trainingId: number): string {
    return this.feedbackSharedBy.get(trainingId) || '';
  }

  fetchTrainingRequests(): void {
    const token = this.authService.getToken();
    if (!token) {
      console.warn('No token available for fetching training requests');
      // Don't clear data if token is missing - maintain consistency
      return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    
    this.http.get<TrainingRequest[]>(this.apiService.getUrl('/training-requests/my'), { headers }).subscribe({
      next: (response) => {
        this.trainingRequests = response || [];
      },
      error: (err) => {
        console.error('Failed to fetch training requests:', err);
        console.error('Error details:', {
          status: err.status,
          statusText: err.statusText,
          error: err.error
        });
        // Don't clear existing data on error - maintain consistency
        // Only clear if unauthorized - silently redirect without showing error
        if (err.status === 401 || err.status === 403) {
          this.authService.logout();
          this.router.navigate(['/login']);
        }
      }
    });
  }

  get myTrainings(): TrainingDetail[] {
    // Use backend-fetched trainings instead of client-side filtering
    // This ensures trainers can see all their trainings even if trainer_name doesn't match exactly
    return this.myTrainingsFromBackend;
  }

  // Getter to determine if user should see trainer zone tab
  // Shows tab if: 1) isTrainer flag is true from API, OR 2) user has trainings where they are listed as trainer
  get shouldShowTrainerZone(): boolean {
    return this.isTrainer || this.myTrainings.length > 0;
  }

  isUpcoming(dateStr?: string): boolean {
    if (!dateStr) return false;
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return new Date(dateStr) >= today;
  }

  getTrainingStatus(training: TrainingDetail): 'upcoming' | 'today' | 'past' | 'no-date' {
    if (!training.training_date) return 'no-date';
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const trainingDate = new Date(training.training_date);
    trainingDate.setHours(0, 0, 0, 0);
    
    if (trainingDate.getTime() === today.getTime()) return 'today';
    if (trainingDate > today) return 'upcoming';
    return 'past';
  }

  getTrainingStatusBadge(status: 'upcoming' | 'today' | 'past' | 'no-date'): { text: string; class: string; icon: string } {
    switch (status) {
      case 'upcoming':
        return { text: 'Upcoming', class: 'bg-amber-100 text-amber-800 ring-amber-200', icon: 'fa-clock' };
      case 'today':
        return { text: 'Today', class: 'bg-green-100 text-green-800 ring-green-200', icon: 'fa-calendar-day' };
      case 'past':
        return { text: 'Past', class: 'bg-slate-100 text-slate-600 ring-slate-200', icon: 'fa-calendar-check' };
      default:
        return { text: 'TBD', class: 'bg-gray-100 text-gray-600 ring-gray-200', icon: 'fa-question' };
    }
  }

  get filteredTrainings(): TrainingDetail[] {
    let list = [...(this.allTrainings || [])];
    if (this.trainingSearch && this.trainingSearch.trim()) {
      const q = this.trainingSearch.trim().toLowerCase();
      list = list.filter(t =>
        (t.training_name || '').toLowerCase().includes(q) ||
        (t.training_topics || '').toLowerCase().includes(q) ||
        (t.trainer_name || '').toLowerCase().includes(q) ||
        (t.skill || '').toLowerCase().includes(q)
      );
    }
    if (this.trainingSkillFilter !== 'All') {
        list = list.filter(t => t.skill === this.trainingSkillFilter);
    }
    if (this.trainingLevelFilter !== 'All') {
        list = list.filter(t => t.skill_category === this.trainingLevelFilter);
    }
    if (this.trainingDateFilter) {
        list = list.filter(t => t.training_date === this.trainingDateFilter);
    }
    list.sort((a, b) => {
        const dateA = a.training_date ? new Date(a.training_date).getTime() : Infinity;
        const dateB = b.training_date ? new Date(b.training_date).getTime() : Infinity;
        return dateA - dateB;
    });
    return list;
  }

  // Method to get request status for a training
  getRequestStatus(trainingId: number): 'none' | 'pending' | 'approved' | 'rejected' {
    const request = this.trainingRequests.find(req => req.training_id === trainingId);
    return request ? request.status : 'none';
  }

  // Method to get request details for a training
  getRequestDetails(trainingId: number): TrainingRequest | null {
    const request = this.trainingRequests.find(req => req.training_id === trainingId) || null;
    return request;
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

  // --- Trainer Zone Modals & Forms ---
  openScheduleTrainingModal(): void {
    this.newTraining.trainer_name = this.employeeName || this.employeeId;
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

  scheduleTraining(): void {
    // Validate required fields
    if (!this.newTraining.training_name || !this.newTraining.training_name.trim()) {
      this.toastService.warning('Please enter a training name.');
      return;
    }
    
    if (!this.newTraining.trainer_name || !this.newTraining.trainer_name.trim()) {
      this.toastService.warning('Please enter a trainer name.');
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const payload = {
      division: this.newTraining.division || null,
      department: this.newTraining.department || null,
      competency: this.newTraining.competency || null,
      skill: this.newTraining.skill || null,
      training_name: this.newTraining.training_name,
      training_topics: this.newTraining.training_topics || null,
      prerequisites: this.newTraining.prerequisites || null,
      skill_category: this.newTraining.skill_category || null,
      trainer_name: this.newTraining.trainer_name,
      email: this.newTraining.email || null,
      training_date: this.newTraining.training_date || null,
      duration: this.newTraining.duration || null,
      time: this.newTraining.time || null,
      training_type: this.newTraining.training_type || null,
      seats: this.newTraining.seats || null,
      assessment_details: this.newTraining.assessment_details || null
    };

    this.http.post(this.apiService.getUrl('/trainings/'), payload, { headers }).subscribe({
      next: () => {
        this.toastService.success('Training scheduled successfully!');
        this.closeScheduleTrainingModal();
        this.fetchScheduledTrainings();
      },
      error: (err) => {
        console.error('Failed to schedule training:', err);
        if (err.status === 401) {
          this.toastService.error('Your session has expired. Please log in again.');
          this.authService.logout();
          this.router.navigate(['/login']);
        } else if (err.status === 422 && err.error && err.error.detail) {
          const errorDetails = err.error.detail.map((e: any) => `- Field '${e.loc[1]}': ${e.msg}`).join('\n');
          this.toastService.error(`Please correct the following errors:\n${errorDetails}`);
        } else {
          this.toastService.error(`Failed to schedule training. ${err.error?.detail || err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  // --- Trainer Zone: Assignment & Feedback Logic ---
  setTrainerZoneView(view: 'overview' | 'assignmentForm' | 'feedbackForm'): void {
    if (view === 'overview') {
      this.resetNewAssignmentForm();
      this.resetNewFeedbackForm();
    }
    this.trainerZoneView = view;
  }

  openShareAssignment(trainingId: number): void {
    this.resetNewAssignmentForm();
    this.newAssignment.trainingId = trainingId;
    
    // Check shared status and load existing data if available
    const token = this.authService.getToken();
    if (token) {
      const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
      this.http.get(this.apiService.getUrl(`/shared-content/trainer/assignments/${trainingId}`), { headers }).subscribe({
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
            
            const currentUser = this.authService.getUsername() || this.employeeId || '';
            if (response.trainer_username && response.trainer_username !== currentUser) {
              this.toastService.info(`Assignment already shared by your co-trainer (${response.trainer_username}). You can update it below.`);
            } else {
              this.toastService.info('Loading existing assignment. You can update it below.');
            }
          } else {
            this.sharedAssignments.set(trainingId, false);
          }
          this.setTrainerZoneView('assignmentForm');
        },
        error: (err) => {
          // If 403, check if it's because assignment exists
          if (err.status === 403) {
            this.toastService.warning('Unable to check assignment status. Please try again.');
          }
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
      this.http.get(this.apiService.getUrl(`/shared-content/trainer/feedback/${trainingId}`), { headers }).subscribe({
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
            
            const currentUser = this.authService.getUsername() || this.employeeId || '';
            if (response.trainer_username && response.trainer_username !== currentUser) {
              this.toastService.info(`Feedback already shared by your co-trainer (${response.trainer_username}). You can update it below.`);
            } else {
              this.toastService.info('Loading existing feedback. You can update it below.');
            }
          } else {
            this.sharedFeedback.set(trainingId, false);
          }
          this.setTrainerZoneView('feedbackForm');
        },
        error: (err) => {
          // If 403, check if it's because feedback exists
          if (err.status === 403) {
            this.toastService.warning('Unable to check feedback status. Please try again.');
          }
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
      const errorMsg = 'Please select a training, provide a title, and add at least one question.';
      this.toastService.error(errorMsg);
      return;
    }

    for (const q of this.newAssignment.questions) {
      if (!q.text.trim()) {
        const errorMsg = 'Please ensure all questions have text.';
        this.toastService.error(errorMsg);
        return;
      }
      if (q.type === 'single-choice' || q.type === 'multiple-choice') {
        if (q.options.some(opt => !opt.text.trim())) {
          const errorMsg = 'Please ensure all options have text.';
          this.toastService.error(errorMsg);
          return;
        }
        if (!q.options.some(opt => opt.isCorrect)) {
          const errorMsg = `Please mark at least one correct answer for the question: "${q.text}"`;
          this.toastService.error(errorMsg);
          return;
        }
      }
    }

    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const payload = {
      training_id: this.newAssignment.trainingId,
      title: this.newAssignment.title,
      description: this.newAssignment.description || '',
      questions: this.newAssignment.questions
    };

    this.http.post(this.apiService.getUrl('/shared-content/assignments'), payload, { headers }).subscribe({
      next: (response: any) => {
        // Mark assignment as shared for this training
        if (this.newAssignment.trainingId) {
          this.sharedAssignments.set(this.newAssignment.trainingId, true);
          // Store who shared it
          if (response.trainer_username) {
            this.assignmentSharedBy.set(this.newAssignment.trainingId, response.trainer_username);
          }
        }
        this.toastService.success('Assignment shared successfully!');
        this.resetNewAssignmentForm();
        this.setTrainerZoneView('overview');
      },
      error: (err) => {
        console.error('Failed to share assignment:', err);
        console.error('Error details:', {
          status: err.status,
          statusText: err.statusText,
          error: err.error,
          message: err.message
        });
        if (err.status === 403) {
          // Check if assignment already exists (shared by co-trainer)
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
      this.toastService.error('Please select a training for the feedback form.');
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
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
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

    this.http.post(this.apiService.getUrl('/shared-content/feedback'), payload, { headers }).subscribe({
      next: (response: any) => {
        // Mark feedback as shared for this training
        if (this.newFeedback.trainingId) {
          this.sharedFeedback.set(this.newFeedback.trainingId, true);
          // Store who shared it
          if (response.trainer_username) {
            this.feedbackSharedBy.set(this.newFeedback.trainingId, response.trainer_username);
          }
        }
        this.toastService.success('Feedback form shared successfully!');
        this.setTrainerZoneView('overview');
      },
      error: (err) => {
        console.error('Failed to share feedback:', err);
        if (err.status === 403) {
          // Check if feedback already exists
          this.checkSharedStatus(this.newFeedback.trainingId!);
          if (this.isFeedbackShared(this.newFeedback.trainingId!)) {
            const sharedBy = this.getFeedbackSharedBy(this.newFeedback.trainingId!);
            this.toastService.warning(`Feedback has already been shared for this training by your co-trainer (${sharedBy}).`);
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

  // --- Skills Modal Logic ---
  openSkillsModal(filterStatus: 'all' | 'Met'): void {
    // Reset modal data first
    this.modalTitle = '';
    this.modalSkills = [];
    
    if (filterStatus === 'all') {
      this.modalTitle = 'Core Skills';
      // For Core Skills, we show all core skills without status
      this.modalSkills = this.sections.map((section, index) => ({
        id: index + 1,
        skill: section.title,
        competency: section.subtitle || 'Core Competency'
      }));
    } else if (filterStatus === 'Met') {
      this.modalTitle = 'Skills Met';
      // For Skills Met, we show the API skills with status "Met"
      this.modalSkills = this.skills.filter(s => s.status === 'Met').map(skill => ({
        id: skill.id,
        skill: skill.skill,
        competency: skill.competency,
        current_expertise: skill.current_expertise,
        target_expertise: skill.target_expertise,
        status: skill.status
      }));
    }
    
    // Force change detection and then show modal
    this.cdr.detectChanges();
    this.showSkillsModal = true;
  }

  closeSkillsModal(): void {
    this.showSkillsModal = false;
    // Reset modal data
    this.modalTitle = '';
    this.modalSkills = [];
  }

  // --- Filter Reset Logic ---
  resetSkillFilters(): void {
    this.skillSearch = '';
    this.skillNameFilter = '';
    this.skillStatusFilter = '';
  }

  resetTrainingFilters(): void {
    this.trainingSearch = '';
    this.trainingSkillFilter = 'All';
    this.trainingLevelFilter = 'All';
    this.trainingDateFilter = '';
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

  // --- User Actions ---
  enrollInTraining(training: TrainingDetail): void {
    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const requestData = { training_id: training.id };

    this.http.post<TrainingRequest>(this.apiService.getUrl('/training-requests/'), requestData, { headers }).subscribe({
      next: (response) => {
        this.toastService.success(`Training request submitted successfully! Your manager will review your request for "${training.training_name}".`);
        this.fetchTrainingRequests(); // Refresh the requests list
      },
      error: (err) => {
        console.error('Failed to submit training request:', err);
        if (err.status === 400) {
          this.toastService.warning(err.error?.detail || 'You have already requested this training.');
        } else if (err.status === 404) {
          this.toastService.error('No manager found for your account. Please contact HR.');
        } else {
          this.toastService.error(`Failed to submit training request. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  viewedAssignment: Assignment | null = null;
  viewedFeedback: { defaultQuestions: FeedbackQuestion[], customQuestions: FeedbackQuestion[], trainingId?: number, sharedFeedbackId?: number } | null = null;
  currentFeedbackTrainingId: number | null = null; // Store training ID separately for feedback submission
  feedbackResponses: Map<number, string> = new Map(); // Store feedback responses: questionIndex -> selectedOption
  showAssignmentModal: boolean = false;
  showFeedbackModal: boolean = false;
  showExamModal: boolean = false;
  userAnswers: UserAnswer[] = [];
  assignmentResult: AssignmentResult | null = null;
  showResultModal: boolean = false;
  isSubmittingAssignment: boolean = false;
  currentExamAssignment: Assignment | null = null;

  viewAssignment(training: TrainingDetail): void {
    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get(this.apiService.getUrl(`/shared-content/assignments/${training.id}`), { headers }).subscribe({
      next: (response: any) => {
        if (response) {
          this.viewedAssignment = {
            trainingId: response.training_id,
            title: response.title,
            description: response.description || '',
            questions: response.questions
          };
          this.showAssignmentModal = true;
        } else {
          this.toastService.warning('No assignment has been shared for this training yet.');
        }
      },
      error: (err) => {
        console.error('Failed to fetch assignment:', err);
        if (err.status === 401) {
          // Silently redirect to login on token expiration
          this.authService.logout();
          this.router.navigate(['/login']);
          return;
        } else if (err.status === 403) {
          this.toastService.error('You can only access assignments for trainings assigned to you.');
        } else if (err.status === 404) {
          this.toastService.warning('No assignment has been shared for this training yet.');
        } else {
          this.toastService.error(`Failed to fetch assignment. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  giveFeedback(training: TrainingDetail): void {
    // Check if attendance has been marked and employee attended
    if (!this.isAttendanceAttended(training.id)) {
      if (!this.isAttendanceMarked(training.id)) {
        this.toastService.warning('Attendance has not been marked by the trainer yet. Please wait for the trainer to mark attendance before giving feedback.');
      } else {
        this.toastService.warning('You were marked as absent for this training. Only employees who attended can give feedback.');
      }
      return;
    }

    // If already submitted, just show a message
    if (this.isFeedbackSubmitted(training.id)) {
      this.toastService.info('You have already submitted feedback for this training.');
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get(this.apiService.getUrl(`/shared-content/feedback/${training.id}`), { headers }).subscribe({
      next: (response: any) => {
        if (response) {
          this.viewedFeedback = {
            defaultQuestions: response.defaultQuestions || [],
            customQuestions: response.customQuestions || [],
            trainingId: training.id,
            sharedFeedbackId: response.id
          };
          this.currentFeedbackTrainingId = training.id;
          this.feedbackResponses.clear(); // Clear previous responses
          this.showFeedbackModal = true;
        } else {
          this.toastService.warning('No feedback form has been shared for this training yet.');
        }
      },
      error: (err) => {
        console.error('Failed to fetch feedback:', err);
        if (err.status === 401) {
          // Silently redirect to login on token expiration
          this.authService.logout();
          this.router.navigate(['/login']);
          return;
        } else if (err.status === 403) {
          this.toastService.error('You can only access feedback for trainings assigned to you.');
        } else if (err.status === 404) {
          this.toastService.warning('No feedback form has been shared for this training yet.');
        } else {
          this.toastService.error(`Failed to fetch feedback. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  closeAssignmentModal(): void {
    this.showAssignmentModal = false;
    this.viewedAssignment = null;
  }

  closeFeedbackModal(): void {
    this.showFeedbackModal = false;
    this.viewedFeedback = null;
    this.currentFeedbackTrainingId = null;
    this.feedbackResponses.clear();
  }

  onFeedbackOptionSelect(questionIndex: number, option: string, questionText: string): void {
    this.feedbackResponses.set(questionIndex, option);
  }

  getFeedbackResponse(questionIndex: number): string | undefined {
    return this.feedbackResponses.get(questionIndex);
  }

  getDefaultQuestionsLength(): number {
    return this.viewedFeedback?.defaultQuestions?.length || 0;
  }

  submitEngineerFeedback(): void {
    if (!this.currentFeedbackTrainingId || !this.viewedFeedback || !this.viewedFeedback.sharedFeedbackId) {
      this.toastService.error('Unable to submit feedback. Please try again.');
      return;
    }

    // Store reference to avoid repeated null checks
    const feedback = this.viewedFeedback;
    const defaultQuestionsLength = feedback.defaultQuestions?.length || 0;

    // Collect all responses
    const allQuestions = [
      ...(feedback.defaultQuestions || []).map((q, i) => ({ ...q, index: i, isDefault: true })),
      ...(feedback.customQuestions || []).map((q, i) => ({ ...q, index: i + defaultQuestionsLength, isDefault: false }))
    ];

    const responses = allQuestions
      .map((question, questionIndex) => {
        const selectedOption = this.feedbackResponses.get(questionIndex);
        if (!selectedOption) {
          return null; // Skip unanswered questions
        }
        return {
          questionIndex: questionIndex,
          questionText: question.text,
          selectedOption: selectedOption
        };
      })
      .filter(r => r !== null) as { questionIndex: number; questionText: string; selectedOption: string }[];

    // Check if all questions are answered
    if (responses.length < allQuestions.length) {
      this.toastService.warning('Please answer all questions before submitting.');
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const payload = {
      training_id: this.currentFeedbackTrainingId,
      shared_feedback_id: this.viewedFeedback.sharedFeedbackId,
      responses: responses
    };

    this.http.post(this.apiService.getUrl('/shared-content/feedback/submit'), payload, { headers }).subscribe({
      next: (response: any) => {
        this.feedbackSubmissionStatus.set(this.currentFeedbackTrainingId!, true);
        this.closeFeedbackModal();
        this.toastService.success('Feedback submitted successfully!');
      },
      error: (err) => {
        console.error('Failed to submit feedback:', err);
        if (err.status === 401) {
          // Silently redirect to login on token expiration
          this.authService.logout();
          this.router.navigate(['/login']);
          return;
        } else if (err.status === 403) {
          this.toastService.error('You can only submit feedback for trainings assigned to you.');
        } else {
          this.toastService.error(`Failed to submit feedback. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  private initializeExam(response: any): void {
    this.currentExamAssignment = {
      trainingId: response.training_id,
      title: response.title,
      description: response.description || '',
      questions: response.questions,
      sharedAssignmentId: response.id
    };
    // Initialize user answers
    this.userAnswers = response.questions.map((q: AssignmentQuestion, index: number) => ({
      questionIndex: index,
      type: q.type,
      selectedOptions: [],
      textAnswer: ''
    }));
    this.showExamModal = true;
  }

  takeExam(training: TrainingDetail): void {
    // Check if attendance has been marked and employee attended
    if (!this.isAttendanceAttended(training.id)) {
      if (!this.isAttendanceMarked(training.id)) {
        this.toastService.warning('Attendance has not been marked by the trainer yet. Please wait for the trainer to mark attendance before taking the assignment.');
      } else {
        this.toastService.warning('You were marked as absent for this training. Only employees who attended can take assignments.');
      }
      return;
    }

    // Check if already submitted - prevent retaking
    if (this.isAssignmentSubmitted(training.id)) {
      this.toastService.warning('You have already submitted this assignment. Click "View Results" to see your score.');
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get(this.apiService.getUrl(`/shared-content/assignments/${training.id}`), { headers }).subscribe({
      next: (response: any) => {
        if (response) {
          // Double-check submission status from backend
          this.http.get(this.apiService.getUrl(`/shared-content/assignments/${training.id}/result`), { headers }).subscribe({
            next: (result: any) => {
              if (result) {
                this.assignmentSubmissionStatus.set(training.id, true);
                this.assignmentScores.set(training.id, result.score || 0);
                // Block retaking if already submitted
                this.toastService.warning('You have already submitted this assignment. Click "View Results" to see your score.');
                return;
              }
              // Initialize exam
              this.initializeExam(response);
            },
            error: (err) => {
              // 403 means attendance not marked - expected, don't log
              // 404 means not submitted yet - also expected
              if (err.status === 403 || err.status === 404) {
                // Attendance not marked or not submitted - can take exam
                this.initializeExam(response);
                return;
              }
              if (err.status === 401) {
                // Silently redirect to login on token expiration
                this.authService.logout();
                this.router.navigate(['/login']);
                return;
              }
              // No result found (404), can take exam
              this.initializeExam(response);
            }
          });
        } else {
          this.toastService.warning('No assignment has been shared for this training yet.');
        }
      },
      error: (err) => {
        console.error('Failed to fetch assignment:', err);
        if (err.status === 401) {
          // Silently redirect to login on token expiration
          this.authService.logout();
          this.router.navigate(['/login']);
          return;
        } else if (err.status === 403) {
          this.toastService.error('You can only access assignments for trainings assigned to you.');
        } else if (err.status === 404) {
          this.toastService.warning('No assignment has been shared for this training yet.');
        } else {
          this.toastService.error(`Failed to fetch assignment. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  closeExamModal(): void {
    this.showExamModal = false;
    this.currentExamAssignment = null;
    this.userAnswers = [];
  }

  onAnswerSelect(questionIndex: number, optionIndex: number, isMultiple: boolean): void {
    if (!this.userAnswers[questionIndex]) {
      return;
    }

    if (isMultiple) {
      // Multiple choice: toggle option
      const currentIndex = this.userAnswers[questionIndex].selectedOptions.indexOf(optionIndex);
      if (currentIndex === -1) {
        this.userAnswers[questionIndex].selectedOptions.push(optionIndex);
      } else {
        this.userAnswers[questionIndex].selectedOptions.splice(currentIndex, 1);
      }
    } else {
      // Single choice: replace selection
      this.userAnswers[questionIndex].selectedOptions = [optionIndex];
    }
  }

  onTextAnswerChange(questionIndex: number, text: string): void {
    if (this.userAnswers[questionIndex]) {
      this.userAnswers[questionIndex].textAnswer = text;
    }
  }

  isOptionSelected(questionIndex: number, optionIndex: number): boolean {
    return this.userAnswers[questionIndex]?.selectedOptions.includes(optionIndex) || false;
  }

  submitExam(): void {
    if (!this.currentExamAssignment || !this.currentExamAssignment.sharedAssignmentId) {
      this.toastService.error('Invalid assignment data.');
      return;
    }

    // Validate all questions are answered
    for (let i = 0; i < this.userAnswers.length; i++) {
      const answer = this.userAnswers[i];
      if (answer.type === 'single-choice' || answer.type === 'multiple-choice') {
        if (answer.selectedOptions.length === 0) {
          this.toastService.warning(`Please answer question ${i + 1}.`);
          return;
        }
      } else if (answer.type === 'text-input') {
        if (!answer.textAnswer || answer.textAnswer.trim() === '') {
          this.toastService.warning(`Please answer question ${i + 1}.`);
          return;
        }
      }
    }

    this.isSubmittingAssignment = true;
    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.isSubmittingAssignment = false;
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const payload = {
      training_id: this.currentExamAssignment.trainingId,
      shared_assignment_id: this.currentExamAssignment.sharedAssignmentId,
      answers: this.userAnswers
    };

    this.http.post(this.apiService.getUrl('/shared-content/assignments/submit'), payload, { headers }).subscribe({
      next: (response: any) => {
        this.isSubmittingAssignment = false;
        this.assignmentResult = {
          id: response.id,
          training_id: response.training_id,
          score: response.score,
          total_questions: response.total_questions,
          correct_answers: response.correct_answers,
          question_results: response.question_results,
          submitted_at: response.submitted_at
        };
        // Mark assignment as submitted and store score
        if (this.currentExamAssignment && this.currentExamAssignment.trainingId) {
          this.assignmentSubmissionStatus.set(this.currentExamAssignment.trainingId, true);
          this.assignmentScores.set(this.currentExamAssignment.trainingId, response.score);
        }
        this.showExamModal = false;
        this.showResultModal = true;
        this.toastService.success(`Assignment submitted! Your score: ${response.score}%`);
      },
      error: (err) => {
        this.isSubmittingAssignment = false;
        console.error('Failed to submit assignment:', err);
        if (err.status === 401) {
          // Silently redirect to login on token expiration
          this.authService.logout();
          this.router.navigate(['/login']);
          return;
        } else if (err.status === 403) {
          this.toastService.error('You can only submit assignments for trainings assigned to you.');
        } else {
          this.toastService.error(`Failed to submit assignment. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  viewAssignmentResult(training: TrainingDetail): void {
    // Check if attendance has been marked and employee attended
    // This is a safety check even though results should only be visible after submission
    if (!this.isAttendanceAttended(training.id)) {
      if (!this.isAttendanceMarked(training.id)) {
        this.toastService.warning('Attendance has not been marked by the trainer yet. Please wait for the trainer to mark attendance.');
      } else {
        this.toastService.warning('You were marked as absent for this training. Only employees who attended can view results.');
      }
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      // Silently redirect to login on token expiration
      this.authService.logout();
      this.router.navigate(['/login']);
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get(this.apiService.sharedAssignmentResultUrl(training.id), { headers }).subscribe({
      next: (response: any) => {
        if (response) {
          // Also fetch assignment to show questions
          this.http.get(this.apiService.getUrl(`/shared-content/assignments/${training.id}`), { headers }).subscribe({
            next: (assignmentResponse: any) => {
              if (assignmentResponse) {
                this.currentExamAssignment = {
                  trainingId: assignmentResponse.training_id,
                  title: assignmentResponse.title,
                  description: assignmentResponse.description || '',
                  questions: assignmentResponse.questions,
                  sharedAssignmentId: assignmentResponse.id
                };
              }
              this.assignmentResult = {
                id: response.id,
                training_id: response.training_id,
                score: response.score,
                total_questions: response.total_questions,
                correct_answers: response.correct_answers,
                question_results: response.question_results,
                submitted_at: response.submitted_at
              };
              this.showResultModal = true;
            },
            error: (err) => {
              if (err.status === 401) {
                // Silently redirect to login on token expiration
                this.authService.logout();
                this.router.navigate(['/login']);
                return;
              }
              // Still show result even if assignment fetch fails
              this.assignmentResult = {
                id: response.id,
                training_id: response.training_id,
                score: response.score,
                total_questions: response.total_questions,
                correct_answers: response.correct_answers,
                question_results: response.question_results,
                submitted_at: response.submitted_at
              };
              this.showResultModal = true;
            }
          });
        } else {
          this.toastService.warning('You have not submitted this assignment yet. Please take the assignment first.');
        }
      },
      error: (err) => {
        console.error('Failed to fetch assignment result:', err);
        if (err.status === 401) {
          // Silently redirect to login on token expiration
          this.authService.logout();
          this.router.navigate(['/login']);
          return;
        } else if (err.status === 404) {
          this.toastService.warning('You have not submitted this assignment yet. Please take the assignment first.');
        } else {
          this.toastService.error(`Failed to fetch results. Error: ${err.statusText || 'Unknown error'}`);
        }
      }
    });
  }

  closeResultModal(): void {
    this.showResultModal = false;
    this.assignmentResult = null;
    this.currentExamAssignment = null;
  }

  getQuestionResult(questionIndex: number): QuestionResult | null {
    if (!this.assignmentResult || !this.assignmentResult.question_results) {
      return null;
    }
    return this.assignmentResult.question_results[questionIndex] || null;
  }

  highlightUpcomingTrainings(): void {
    const element = document.getElementById('upcoming-trainings-section');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });

      // Directly apply styles for a more reliable effect
      element.style.transition = 'box-shadow 0.5s ease-in-out';
      element.style.boxShadow = '0 0 0 4px #38bdf8, 0 0 15px #0ea5e9'; // A blue glow, similar to a ring

      // Remove the styles after a delay
      setTimeout(() => {
        element.style.boxShadow = 'none';
      }, 2500);
    } else {
      console.error("DEBUG: Could not find element with id 'upcoming-trainings-section'");
    }
  }

  // --- General Helpers ---
  getFilteredSkills(): Skill[] {
    let filtered = this.skills;
    if (this.skillNameFilter) {
      filtered = filtered.filter(skill => skill.skill === this.skillNameFilter);
    }
    return filtered;
  }

  getFilteredRequests(): TrainingRequest[] {
    let filtered = this.trainingRequests;
    
    // Filter by status
    if (this.requestStatusFilter && this.requestStatusFilter !== 'all') {
      filtered = filtered.filter(request => request.status === this.requestStatusFilter);
    }
    
    // Filter by search term
    if (this.requestSearch.trim()) {
      const searchTerm = this.requestSearch.trim().toLowerCase();
      filtered = filtered.filter(request => 
        request.training.training_name.toLowerCase().includes(searchTerm) ||
        request.training.trainer_name.toLowerCase().includes(searchTerm) ||
        (request.training.skill && request.training.skill.toLowerCase().includes(searchTerm)) ||
        (request.training.skill_category && request.training.skill_category.toLowerCase().includes(searchTerm))
      );
    }
    
    return filtered;
  }

  getSkillProgress(competency: Skill | ModalSkill): number {
    // Calculate progress based on average overall performance across L1-L5 levels
    const skillName = competency.skill;
    if (!skillName) return 0;

    // Get all feedback for this skill
    const feedbackList = this.getFeedbackForSkill(skillName);
    if (!feedbackList || feedbackList.length === 0) return 0;

    // Group feedback by skill category (L1, L2, L3, L4, L5)
    const feedbackByLevel = new Map<string, any[]>();
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

    // If no performance scores found, return 0
    if (performanceScores.length === 0) return 0;

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

  getFormattedDate(): string {
    return new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  }

  selectTab(tabName: string): void {
    this.activeTab = tabName;
    // Refresh data for each tab to maintain consistency after logout/login
    if (tabName === 'dashboard') {
      // Refresh dashboard data and all related data
      this.fetchDashboardData();
      this.fetchAssignedTrainings();
      // Refresh manager feedback to show latest updates
      this.fetchManagerFeedback();
    }
    if (tabName === 'mySkills') {
      // Refresh skills data
      this.fetchDashboardData();
      this.loadAdditionalSkills();
    }
    if (tabName === 'trainingCatalog') {
      // Refresh training catalog and requests
      this.fetchScheduledTrainings();
      this.fetchTrainingRequests();
    }
    if (tabName === 'assignedTrainings') {
      // Refresh assigned trainings
      this.fetchAssignedTrainings();
      // Refresh manager feedback when viewing assigned trainings
      this.fetchManagerFeedback();
    }
    if (tabName === 'myRequests') {
      // Refresh training requests
      this.fetchTrainingRequests();
    }
    if (tabName === 'trainerZone') {
      // Refresh scheduled trainings when switching to Trainer Zone to show latest sessions
      this.fetchScheduledTrainings();
    }
  }

  logout(): void {
    // Clear notifications before logging out
    this.notificationService.clear();
    // Clear only authentication data (not component data)
    // Component will be destroyed, and data will be fresh when user logs back in
    // This ensures data consistency - user will see current data from backend on next login
    this.authService.logout();
    this.router.navigate(['/login']);
  }

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

  getFilteredSections(): Section[] {
    let sectionsToFilter = this.sections;
    if (this.selectedSkill) {
      sectionsToFilter = this.sections.filter(sec => sec.title === this.selectedSkill);
    }
    const q = this.levelsSearch.trim().toLowerCase();
    if (!q) return sectionsToFilter;

    return sectionsToFilter.map(sec => {
      const matchTitle = sec.title.toLowerCase().includes(q) || (sec.subtitle ?? '').toLowerCase().includes(q);
      const filteredLevels = sec.levels
        .map(l => ({ ...l, items: l.items.filter(it => it.toLowerCase().includes(q)) }))
        .filter(l => l.items.length > 0);
      
      // If search query matches a skill title, show all its levels
      // Otherwise, only show levels that have matching items
      if (matchTitle) {
          return sec;
      }
      if (filteredLevels.length > 0) {
          return { ...sec, levels: filteredLevels };
      }
      return null;
    }).filter((s): s is Section => s !== null);
  }

  onSkillChange(): void {}

  // --- Visual Helpers ---
  getLevelHeaderClass = (level: number) => ['bg-red-50', 'bg-orange-50', 'bg-yellow-50', 'bg-blue-50', 'bg-green-50'][level - 1] || 'bg-gray-50';
  getLevelBadgeClass = (level: number) => ['bg-sky-500', 'bg-orange-500', 'bg-yellow-500', 'bg-blue-500', 'bg-emerald-500'][level - 1] || 'bg-gray-500';
  getLevelTitle = (level: number) => ['Beginner', 'Basic', 'Intermediate', 'Advanced', 'Expert'][level - 1] || 'Unknown';
  getLevelIcon = (level: number) => ['fa-solid fa-seedling text-sky-500', 'fa-solid fa-leaf text-sky-500', 'fa-solid fa-tree text-sky-600', 'fa-solid fa-rocket text-sky-500', 'fa-solid fa-crown text-sky-500'][level - 1] || 'fa-solid fa-circle';
  getComplexityDots = (level: number) => Array.from({ length: 5 }, (_, i) => i < level);
  getProgressBarClass = () => this.progressPercentage >= 80 ? 'bg-gradient-to-r from-green-400 to-green-600' : this.progressPercentage >= 60 ? 'bg-gradient-to-r from-sky-400 to-sky-600' : this.progressPercentage >= 40 ? 'bg-gradient-to-r from-yellow-400 to-yellow-600' : 'bg-gradient-to-r from-orange-400 to-orange-600';

  // --- Category & Gap Logic ---
  getSkillCategory(skillName: string): string {
    const skill = skillName.toLowerCase();
    if (['c++', 'cpp', 'python', 'programming'].some(kw => skill.includes(kw))) return 'Programming';
    if (['git', 'version control'].some(kw => skill.includes(kw))) return 'Version Control';
    if (['test', 'exam', 'axivion'].some(kw => skill.includes(kw))) return 'Testing & Quality';
    if (['azure', 'devops'].some(kw => skill.includes(kw))) return 'DevOps';
    if (['doors', 'integrity', 'softcar', 'matlab'].some(kw => skill.includes(kw))) return 'Engineering Tools';
    return 'Technical';
  }

  getTrainingCardIcon(skill?: string): string {
    if (!skill) return 'fa-solid fa-laptop-code';
    const s = skill.toLowerCase();
    if (s.includes('python')) return 'fa-brands fa-python';
    if (s.includes('c++') || s.includes('cpp')) return 'fa-solid fa-file-code';
    if (s.includes('git')) return 'fa-brands fa-git-alt';
    if (s.includes('azure')) return 'fa-brands fa-microsoft';
    if (s.includes('exam') || s.includes('axivion')) return 'fa-solid fa-vial-circle-check';
    return 'fa-solid fa-laptop-code';
  }

  // --- NEW COUNTERS FOR DASHBOARD WIDGETS ---
  getAdditionalSkillsTotalCount = () => this.additionalSkills.length;
  getAdditionalTechnicalSkillsCount = () => this.additionalSkills.filter(s => s.skill_category === 'Technical').length;
  getAdditionalSoftSkillsCount = () => this.additionalSkills.filter(s => s.skill_category === 'Soft Skills').length;
  getAdditionalLeadershipSkillsCount = () => this.additionalSkills.filter(s => s.skill_category === 'Leadership').length;

  // --- Additional Skills CRUD ---
  loadAdditionalSkills(): void {
    const token = this.authService.getToken();
    if (!token) {
      console.warn('No token available for fetching additional skills');
      // Don't clear data if token is missing - maintain consistency
      return;
    }
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get<any[]>(this.apiService.getUrl('/additional-skills/'), { headers }).subscribe({
      next: (skills) => { 
        this.additionalSkills = skills || [];
      },
      error: (err) => {
        console.error('Failed to fetch additional skills:', err);
        // Don't clear existing data on error - maintain consistency
        // Only redirect if unauthorized - silently redirect without showing error (data will be refreshed on next login)
        if (err.status === 401 || err.status === 403) {
          this.authService.logout();
          this.router.navigate(['/login']);
        }
        // For other errors, keep existing data
      }
    });
  }

  toggleAddSkillForm(): void {
    this.showAddSkillForm = !this.showAddSkillForm;
    if (!this.showAddSkillForm) this.resetNewSkillForm();
  }

  addNewSkill(): void {
    if (!this.newSkill.name.trim()) return;
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    const skillData = {
      skill_name: this.newSkill.name.trim(),
      skill_level: this.newSkill.level,
      skill_category: this.newSkill.category,
      description: this.newSkill.description.trim() || null
    };

    const request = this.editingSkillId
      ? this.http.put<any>(this.apiService.getUrl(`/additional-skills/${this.editingSkillId}`), skillData, { headers })
      : this.http.post<any>(this.apiService.getUrl('/additional-skills/'), skillData, { headers });

    request.subscribe({
      next: (savedSkill) => {
        if (this.editingSkillId) {
          const index = this.additionalSkills.findIndex(s => s.id === this.editingSkillId);
          if (index !== -1) this.additionalSkills[index] = savedSkill;
        } else {
          this.additionalSkills.push(savedSkill);
        }
        this.resetNewSkillForm();
        this.showAddSkillForm = false;
      }
    });
  }

  removeAdditionalSkill(skillId: number): void {
    const token = this.authService.getToken();
    if (!token) return;
    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.delete(this.apiService.getUrl(`/additional-skills/${skillId}`), { headers }).subscribe({
      next: () => { this.additionalSkills = this.additionalSkills.filter(skill => skill.id !== skillId); }
    });
  }

  editAdditionalSkill(skill: any): void {
    this.newSkill = { name: skill.skill_name, level: skill.skill_level, category: skill.skill_category, description: skill.description || '' };
    this.showAddSkillForm = true;
    this.editingSkillId = skill.id;
  }

  resetNewSkillForm(): void {
    this.newSkill = { name: '', level: 'Beginner', category: 'Technical', description: '' };
    this.editingSkillId = null;
  }

  getSkillLevelColor = (level: string) => ({
    'Beginner': 'bg-gray-100 text-gray-700 border border-gray-300',
    'Intermediate': 'bg-sky-100 text-sky-700 border border-sky-300',
    'Advanced': 'bg-violet-100 text-violet-700 border border-violet-300',
    'Expert': 'bg-amber-100 text-amber-700 border border-amber-300',
  }[level] || 'bg-gray-100 text-gray-700 border border-gray-300');

  getCategoryColor = (category: string) => ({
    'Technical': 'bg-slate-100 text-slate-700 border border-slate-300',
    'Soft Skills': 'bg-stone-100 text-stone-700 border border-stone-300',
    'Leadership': 'bg-zinc-100 text-zinc-700 border border-zinc-300',
    'Communication': 'bg-neutral-100 text-neutral-700 border border-neutral-300',
    'Project Management': 'bg-gray-100 text-gray-700 border border-gray-300',
  }[category] || 'bg-gray-100 text-gray-700 border border-gray-300');

  // --- Training File Management ---
  questionFilesUploaded: Map<number, boolean> = new Map(); // Track which trainings have question files
  solutionFilesUploaded: Map<number, boolean> = new Map(); // Track which trainings have solution files uploaded by engineer
  trainerSolutions: Map<number, any[]> = new Map(); // Store solution files for trainers to view

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

  downloadQuestionFile(trainingId: number): void {
    // Check if attendance has been marked and employee attended
    if (!this.isAttendanceAttended(trainingId)) {
      if (!this.isAttendanceMarked(trainingId)) {
        this.toastService.warning('Attendance has not been marked by the trainer yet. Please wait for the trainer to mark attendance before downloading questions.');
      } else {
        this.toastService.warning('You were marked as absent for this training. Only employees who attended can download questions.');
      }
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      return;
    }

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get(this.apiService.questionFileUrl(trainingId), { 
      headers, 
      responseType: 'blob' 
    }).subscribe({
      next: (blob: Blob) => {
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `training_questions_${trainingId}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        this.toastService.success('Question file downloaded successfully');
      },
      error: (err) => {
        console.error('Failed to download question file:', err);
        this.toastService.error(err.error?.detail || 'Failed to download question file');
      }
    });
  }

  uploadSolutionFile(trainingId: number, event: any): void {
    // Check if attendance has been marked and employee attended
    if (!this.isAttendanceAttended(trainingId)) {
      if (!this.isAttendanceMarked(trainingId)) {
        this.toastService.warning('Attendance has not been marked by the trainer yet. Please wait for the trainer to mark attendance before uploading solutions.');
      } else {
        this.toastService.warning('You were marked as absent for this training. Only employees who attended can upload solutions.');
      }
      event.target.value = ''; // Reset file input
      return;
    }

    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      this.toastService.error('Only PDF files are allowed');
      event.target.value = ''; // Reset file input
      return;
    }

    const token = this.authService.getToken();
    if (!token) {
      this.toastService.error('Authentication error. Please log in again.');
      event.target.value = ''; // Reset file input
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('training_id', trainingId.toString());

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });

    this.http.post(this.apiService.uploadSolutionFileUrl, formData, { headers }).subscribe({
      next: (response: any) => {
        this.toastService.success('Solution file uploaded successfully');
        this.solutionFilesUploaded.set(trainingId, true);
        event.target.value = ''; // Reset file input
      },
      error: (err) => {
        console.error('Failed to upload solution file:', err);
        this.toastService.error(err.error?.detail || 'Failed to upload solution file');
      }
    });
  }

  fetchTrainerSolutions(trainingId: number): void {
    const token = this.authService.getToken();
    if (!token) return;

    const headers = new HttpHeaders({ 'Authorization': `Bearer ${token}` });
    this.http.get<any[]>(this.apiService.trainerSolutionsUrl(trainingId), { headers }).subscribe({
      next: (solutions) => {
        this.trainerSolutions.set(trainingId, solutions || []);
      },
      error: (err) => {
        console.error('Failed to fetch solutions:', err);
        this.trainerSolutions.set(trainingId, []);
      }
    });
  }

  downloadSolutionFile(trainingId: number, employeeId: string): void {
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
        link.download = `solution_${trainingId}_${employeeId}.pdf`;
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

  hasQuestionFile(trainingId: number): boolean {
    return this.questionFilesUploaded.get(trainingId) || false;
  }

  hasSolutionFile(trainingId: number): boolean {
    return this.solutionFilesUploaded.get(trainingId) || false;
  }

  getTrainingSolutions(trainingId: number): any[] {
    return this.trainerSolutions.get(trainingId) || [];
  }
}