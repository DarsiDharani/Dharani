"""
Email Service Module using pywin32

Purpose: Send email notifications through Microsoft Outlook using pywin32
Features:
- Send emails when training enrollment requests are created
- Send emails when managers approve/reject requests
- Uses Outlook COM interface via pywin32

Requirements:
- Windows OS
- Microsoft Outlook installed and configured
- pywin32 library installed

@author Orbit Skill Development Team
@date 2025
"""

import win32com.client
import pythoncom
from typing import Optional
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
import subprocess
import time
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Thread pool for email sending (COM objects need to run in separate threads)
_email_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="email")

class OutlookEmailService:
    """
    Service class for sending emails through Microsoft Outlook using pywin32.
    Thread-safe implementation for async FastAPI context.
    """
    
    def __init__(self):
        """Initialize the Outlook email service."""
        self._lock = threading.Lock()
    
    def _get_outlook(self):
        """
        Get or create Outlook COM object.
        Initializes COM for the current thread if needed.
        """
        try:
            # Try to get existing COM object for this thread
            try:
                pythoncom.CoInitialize()
            except pythoncom.com_error:
                # Already initialized in this thread, that's fine
                pass
            
            # Create Outlook application object
            outlook = win32com.client.Dispatch("Outlook.Application")
            return outlook
        except Exception as e:
            logger.error(f"Failed to initialize Outlook: {str(e)}")
            raise Exception(f"Failed to initialize Outlook. Make sure Outlook is installed and configured. Error: {str(e)}")
    
    def _get_email_address(self, username: str, email_from_db: Optional[str] = None) -> str:
        """
        Get email address for a user.
        First tries to use email from database (employee_competency table),
        otherwise falls back to constructing from username.
        
        Args:
            username: Employee username/ID
            email_from_db: Optional email address from employee_competency table
            
        Returns:
            Email address string
        """
        # If email is provided from database, use it
        if email_from_db and email_from_db.strip() and '@' in email_from_db:
            return email_from_db.strip()
        
        # Fallback: construct email from username
        # Default pattern - modify this to match your organization
        # Common patterns:
        # - username@company.com
        # - firstname.lastname@company.com
        # - username@domain.com
        
        return f"{username}@company.com"
    
    def _send_email_sync(
        self,
        to_email: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        is_html: bool = False,
        display_before_send: bool = True  # Default to True so user can review
    ) -> bool:
        """
        Internal method to send email synchronously in a separate thread.
        COM objects must run in the same thread where they're created.
        """
        try:
            # Initialize COM for this thread - use apartment threading for Outlook
            try:
                pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)
            except:
                # Already initialized, that's fine
                pass
            
            try:
                # Create Outlook application object
                logger.info("Initializing Outlook COM object in thread...")
                
                # Try multiple connection methods to connect to Outlook
                outlook = None
                
                # Method 1: Try Dispatch (connects to existing Outlook instance or creates new)
                try:
                    outlook = win32com.client.Dispatch("Outlook.Application")
                    logger.info("✅ Connected to Outlook via Dispatch")
                except Exception as e1:
                    error_code1 = getattr(e1, 'args', [None])[0] if hasattr(e1, 'args') and len(e1.args) > 0 else None
                    logger.info(f"Dispatch failed (code: {error_code1}), trying GetActiveObject...")
                    
                    # Method 2: Try GetActiveObject (for already running instance)
                    try:
                        outlook = win32com.client.GetActiveObject("Outlook.Application")
                        logger.info("✅ Connected to Outlook via GetActiveObject")
                    except Exception as e2:
                        error_code2 = getattr(e2, 'args', [None])[0] if hasattr(e2, 'args') and len(e2.args) > 0 else None
                        logger.warning(f"Both methods failed. Dispatch: {error_code1}, GetActiveObject: {error_code2}")
                        
                        # If "Invalid class string" error, Outlook COM isn't registered/accessible
                        if -2147221005 in [error_code1, error_code2]:
                            logger.warning("⚠️  Outlook COM not accessible")
                            logger.warning("   CRITICAL: Outlook must be opened AFTER starting the Python server!")
                            logger.warning("   Steps:")
                            logger.warning("   1. Stop the backend server (Ctrl+C)")
                            logger.warning("   2. Close Outlook completely (check system tray)")
                            logger.warning("   3. Start the backend server again")
                            logger.warning("   4. Open Outlook")
                            logger.warning("   5. Wait for Outlook to fully load (10-30 seconds)")
                            logger.warning("   6. Then try creating the training request")
                            return False
                        else:
                            logger.warning(f"⚠️  Cannot connect to Outlook")
                            logger.warning("   Make sure Outlook is fully loaded and try again")
                            return False
                
                if not outlook:
                    raise Exception("Failed to create Outlook object")
                
                # Create a new mail item
                mail = outlook.CreateItem(0)  # 0 = olMailItem
                
                # Set recipient
                mail.To = to_email
                
                # Set CC if provided
                if cc:
                    mail.CC = cc
                
                # Set subject
                mail.Subject = subject
                
                # Set body
                if is_html:
                    mail.HTMLBody = body
                else:
                    mail.Body = body
                
                # Log email details
                logger.info(f"📧 Email prepared:")
                logger.info(f"   To: {to_email}")
                logger.info(f"   Subject: {subject}")
                
                # Validate email address format
                if '@' not in to_email:
                    logger.error(f"❌ Invalid email address format: {to_email}")
                    return False
                
                # Display email window for user to review and send manually
                # This is more reliable than auto-send and allows user to verify
                logger.info("📬 Opening email in Outlook window for review and manual send...")
                try:
                    mail.Display(True)  # True = modal window (blocks until closed)
                    logger.info("✅ Email window opened. User can review and send manually.")
                    logger.info("   After sending, check Outlook Sent Items folder.")
                    return True
                except Exception as display_error:
                    logger.warning(f"Display failed: {display_error}, trying Send instead...")
                    # Fallback: try to send directly
                    try:
                        mail.Send()
                        logger.info("✅ Email sent directly (Display failed, but Send succeeded)")
                        return True
                    except Exception as send_error:
                        logger.error(f"Both Display and Send failed: {send_error}")
                        raise
                
            finally:
                # Cleanup COM
                pythoncom.CoUninitialize()
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Failed to send email to {to_email}: {error_msg}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full error details: {repr(e)}")
            
            # Provide helpful error messages
            if "COM" in error_msg or "Dispatch" in error_msg or "CreateObject" in error_msg:
                logger.error("⚠️ Outlook COM error. Make sure:")
                logger.error("   1. Outlook is installed")
                logger.error("   2. Outlook is running (open Outlook application)")
                logger.error("   3. Outlook has at least one email account configured")
            elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                logger.error("⚠️ Permission error. Outlook may require user confirmation.")
            
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            
            return False
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        is_html: bool = False,
        display_before_send: bool = True  # Default to True
    ) -> bool:
        """
        Send an email through Outlook.
        Runs in a separate thread to handle COM objects properly.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body content
            cc: Optional CC email address
            is_html: Whether the body is HTML formatted
            display_before_send: If True, shows email window for manual send (recommended)
            
        Returns:
            True if email window was opened successfully, False otherwise
        """
        try:
            # Submit email sending to thread pool
            future = _email_executor.submit(
                self._send_email_sync,
                to_email,
                subject,
                body,
                cc,
                is_html,
                display_before_send
            )
            
            # Wait for result (with timeout)
            result = future.result(timeout=30)  # 30 second timeout
            return result
            
        except Exception as e:
            logger.error(f"❌ Error submitting email to thread pool: {str(e)}")
            return False
    
    def send_training_request_notification(
        self,
        manager_username: str,
        employee_username: str,
        employee_name: str,
        training_name: str,
        training_id: int,
        manager_email: Optional[str] = None
    ) -> bool:
        """
        Send email notification to manager when an engineer requests training enrollment.
        
        Args:
            manager_username: Manager's username
            employee_username: Employee's username who made the request
            employee_name: Employee's name
            training_name: Name of the training course
            training_id: Training ID
            manager_email: Optional email address from employee_competency table
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        manager_email_addr = self._get_email_address(manager_username, manager_email)
        
        subject = f"Training Enrollment Request - {training_name}"
        
        body = f"""
Dear Manager,

You have received a new training enrollment request from your team member.

Request Details:
- Employee: {employee_name} ({employee_username})
- Training Course: {training_name}
- Training ID: {training_id}
- Request Date: {self._get_current_datetime()}

Please review and respond to this request through the Orbit Skill portal.

Best regards,
Orbit Skill System
        """.strip()
        
        return self.send_email(
            to_email=manager_email_addr,
            subject=subject,
            body=body
        )
    
    def send_request_decision_notification(
        self,
        employee_username: str,
        employee_name: str,
        manager_username: str,
        training_name: str,
        status: str,
        manager_notes: Optional[str] = None,
        employee_email: Optional[str] = None
    ) -> bool:
        """
        Send email notification to employee when manager approves or rejects their request.
        
        Args:
            employee_username: Employee's username
            employee_name: Employee's name
            manager_username: Manager's username
            training_name: Name of the training course
            status: 'approved' or 'rejected'
            manager_notes: Optional notes from manager
            employee_email: Optional email address from employee_competency table
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        employee_email_addr = self._get_email_address(employee_username, employee_email)
        
        status_text = "Approved" if status == "approved" else "Rejected"
        subject = f"Training Request {status_text} - {training_name}"
        
        body = f"""
Dear {employee_name},

Your training enrollment request has been {status_text.lower()}.

Request Details:
- Training Course: {training_name}
- Status: {status_text}
- Decision Date: {self._get_current_datetime()}
        """
        
        if manager_notes:
            body += f"\n- Manager Notes: {manager_notes}"
        
        if status == "approved":
            body += "\n\nCongratulations! You have been enrolled in this training. Please check your dashboard for more details."
        else:
            body += "\n\nIf you have any questions, please contact your manager."
        
        body += "\n\nBest regards,\nOrbit Skill System"
        
        return self.send_email(
            to_email=employee_email_addr,
            subject=subject,
            body=body
        )
    
    def _get_current_datetime(self) -> str:
        """Get current date and time as formatted string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def __del__(self):
        """Cleanup COM initialization when object is destroyed."""
        try:
            pythoncom.CoUninitialize()
        except:
            pass


# Global instance (singleton pattern)
_email_service_instance: Optional[OutlookEmailService] = None

def get_email_service() -> OutlookEmailService:
    """
    Get or create the global email service instance.
    
    Returns:
        OutlookEmailService instance
    """
    global _email_service_instance
    if _email_service_instance is None:
        try:
            _email_service_instance = OutlookEmailService()
        except Exception as e:
            logger.error(f"Failed to create email service: {str(e)}")
            # Return a dummy service that logs but doesn't send
            return DummyEmailService()
    return _email_service_instance


class DummyEmailService:
    """
    Dummy email service that logs but doesn't send emails.
    Used as fallback when Outlook is not available.
    """
    
    def send_training_request_notification(self, *args, **kwargs) -> bool:
        logger.warning("Email service not available. Training request notification not sent.")
        return False
    
    def send_request_decision_notification(self, *args, **kwargs) -> bool:
        logger.warning("Email service not available. Request decision notification not sent.")
        return False
