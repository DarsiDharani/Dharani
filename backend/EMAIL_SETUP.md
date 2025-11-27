# Email Notification Setup Guide

This guide explains how to set up and configure email notifications for training enrollment requests using pywin32 and Microsoft Outlook.

## Prerequisites

1. **Windows OS**: This feature requires Windows as pywin32 is Windows-specific
2. **Microsoft Outlook**: Outlook must be installed and configured on the server
3. **pywin32 Library**: Install using `pip install pywin32`

## Installation

1. Install pywin32:
   ```bash
   pip install pywin32
   ```

2. After installation, run the post-install script (if needed):
   ```bash
   python Scripts/pywin32_postinstall.py -install
   ```

## Configuration

### Email Address Source

The system now retrieves email addresses from the `employee_competency` table in the database. If an email is not found in the database, it falls back to constructing the email using the pattern: `{username}@company.com`

**Important**: Make sure your `employee_competency` table has an `email` column with email addresses for your employees. The email field is automatically loaded when you upload Excel files with an "email" column.

To customize the fallback email address pattern, edit `backend/app/email_service.py` and modify the `_get_email_address` method:

```python
def _get_email_address(self, username: str) -> str:
    """
    Construct email address from username.
    Customize this method to match your organization's email pattern.
    """
    # Example patterns:
    # - username@company.com
    # - firstname.lastname@company.com
    # - username@domain.com
    
    # Customize this line:
    return f"{username}@yourcompany.com"
```

### Alternative: Email Mapping Dictionary

If you have a specific mapping of usernames to email addresses, you can modify the `_get_email_address` method to use a dictionary or database lookup:

```python
def _get_email_address(self, username: str) -> str:
    # Option 1: Use a dictionary
    email_map = {
        "1234567": "john.doe@company.com",
        "7654321": "jane.smith@company.com",
        # Add more mappings as needed
    }
    return email_map.get(username, f"{username}@company.com")
    
    # Option 2: Query from database (if you add email to User model)
    # This would require adding an email field to the User model
```

## How It Works

### When Engineer Requests Training

1. Engineer submits a training enrollment request through the UI
2. System creates the request in the database
3. **Email is automatically sent to the manager** with:
   - Employee name and ID
   - Training course name
   - Request date
   - Link to review the request

### When Manager Approves/Rejects

1. Manager approves or rejects the request through the UI
2. System updates the request status in the database
3. **Email is automatically sent to the employee** with:
   - Decision (approved/rejected)
   - Training course name
   - Manager notes (if provided)
   - Next steps information

## Troubleshooting

### Outlook Not Found Error

If you see errors like "Failed to initialize Outlook":
- Ensure Outlook is installed on the server
- Ensure Outlook is configured with at least one email account
- Try opening Outlook manually to verify it works
- Check that pywin32 is properly installed

### Emails Not Sending

1. Check the application logs for error messages
2. Verify Outlook is running or can be started automatically
3. Ensure the email addresses are correctly formatted
4. Check Outlook's security settings (may prompt for permission)

### Email Address Issues

- Verify the `_get_email_address` method matches your organization's email pattern
- Test with a known username to see what email address is generated
- Consider adding email addresses to the User model for more reliable mapping

## Testing

To test the email functionality:

1. Create a training enrollment request as an engineer
2. Check that the manager receives an email notification
3. As a manager, approve or reject the request
4. Check that the employee receives an email notification

## Notes

- Email sending failures are logged but do not prevent the request from being created/updated
- The system uses Outlook's default email account to send emails
- Emails are sent synchronously, so there may be a slight delay in the API response
- For production, consider implementing a background task queue for email sending

