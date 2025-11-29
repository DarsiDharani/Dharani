# Notification System - All Cases

This document lists all notification cases implemented in the Orbit Skill application.

## Currently Implemented (Active Notifications)

### 1. **Training Assigned to Employee** ✅
- **Trigger**: When a manager assigns a training to an employee
- **Location**: `backend/app/routes/assignment_routes.py` - `assign_training_to_employee()`
- **Recipient**: Employee
- **Notification Type**: `assignment`
- **Title**: "New Training Assigned"
- **Message**: "You have been assigned to the training: {training_name}"
- **Action URL**: `/engineer-dashboard?tab=assignedTrainings`
- **Icon**: Blue assignment icon

### 2. **Training Request Received by Manager** ✅
- **Trigger**: When an employee requests approval for a training
- **Location**: `backend/app/routes/training_requests.py` - `create_training_request()`
- **Recipient**: Manager
- **Notification Type**: `info`
- **Title**: "Training Request Received"
- **Message**: "{employee_name} has requested approval for: {training_name}"
- **Action URL**: `/manager-dashboard?tab=trainingRequests`
- **Icon**: Blue info icon

### 3. **Training Request Approved** ✅
- **Trigger**: When a manager approves an employee's training request
- **Location**: `backend/app/routes/training_requests.py` - `respond_to_request()`
- **Recipient**: Employee
- **Notification Type**: `success`
- **Title**: "Training Request Approved"
- **Message**: "Your request for '{training_name}' has been approved by your manager."
- **Action URL**: `/engineer-dashboard?tab=assignedTrainings`
- **Icon**: Green success icon

### 4. **Training Request Rejected** ✅
- **Trigger**: When a manager rejects an employee's training request
- **Location**: `backend/app/routes/training_requests.py` - `respond_to_request()`
- **Recipient**: Employee
- **Notification Type**: `warning`
- **Title**: "Training Request Rejected"
- **Message**: "Your request for '{training_name}' has been rejected by your manager." (includes manager notes if provided)
- **Action URL**: `/engineer-dashboard?tab=trainingRequests`
- **Icon**: Yellow warning icon

---

### 5. **New Assignment Available** ✅
- **Trigger**: When a trainer shares/creates or updates an assignment for a training
- **Location**: `backend/app/routes/shared_content_routes.py` - `share_assignment()`
- **Recipient**: All employees assigned to that training
- **Notification Type**: `assignment`
- **Title**: "New Assignment Available"
- **Message**: "A new assignment '{assignment_title}' is available for training: {training_name}"
- **Action URL**: `/engineer-dashboard?tab=assignedTrainings`
- **Icon**: Blue assignment icon
- **Note**: Notifies all assigned employees when assignment is created or updated

### 6. **New Feedback Available** ✅
- **Trigger**: When a trainer shares/creates or updates a feedback form for a training
- **Location**: `backend/app/routes/shared_content_routes.py` - `share_feedback()`
- **Recipient**: All employees assigned to that training
- **Notification Type**: `info`
- **Title**: "Feedback Requested"
- **Message**: "Please submit feedback for the training: {training_name}"
- **Action URL**: `/engineer-dashboard?tab=assignedTrainings`
- **Icon**: Blue info icon
- **Note**: Notifies all assigned employees when feedback form is created or updated

### 7. **Performance Feedback Received** ✅
- **Trigger**: When a manager provides performance feedback for an employee
- **Location**: `backend/app/routes/shared_content_routes.py` - `create_or_update_performance_feedback()`
- **Recipient**: Employee
- **Notification Type**: `info`
- **Title**: "Performance Feedback Received"
- **Message**: "Your manager has provided performance feedback for: {training_name}"
- **Action URL**: `/engineer-dashboard?tab=assignedTrainings`
- **Icon**: Blue info icon

---

## Notification Types and Visual Indicators

| Type | Color | Icon | Usage |
|------|-------|------|-------|
| `info` | Blue/Sky | `fa-info-circle` | General information, feedback requests |
| `success` | Green | `fa-check-circle` | Approvals, successful actions |
| `warning` | Yellow | `fa-exclamation-triangle` | Rejections, warnings |
| `error` | Red | `fa-times-circle` | Errors (not currently used) |
| `assignment` | Blue | `fa-clipboard-list` | Training assignments, new assignments |

---

## How to Add New Notification Cases

1. **Create a notification function** in `backend/app/notification_service.py`:
   ```python
   async def notify_your_event(
       db: AsyncSession,
       user_empid: str,
       # ... other parameters
   ) -> Notification:
       return await create_notification(
           db=db,
           user_empid=user_empid,
           title="Your Title",
           message="Your message",
           type="info",  # or success, warning, error, assignment
           related_id=related_id,
           related_type="your_type",
           action_url="/dashboard?tab=relevantTab"
       )
   ```

2. **Call the function** in the appropriate route handler:
   ```python
   from app.notification_service import notify_your_event
   try:
       await notify_your_event(
           db=db,
           user_empid=user_id,
           # ... other parameters
       )
   except Exception as e:
       logging.error(f"Failed to create notification: {str(e)}")
   ```

---

## Summary

- **Total Notification Functions**: 7
- **All Implemented**: 7 ✅
- **Status**: All notification cases are fully implemented and active

All notification functions are designed to be non-blocking - if notification creation fails, it won't break the main operation (assignment, request, etc.). Error handling is in place to log failures without affecting the primary functionality.

