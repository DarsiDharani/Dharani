"""
Test script to verify Outlook email functionality
Run this to test if Outlook email sending is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.email_service import get_email_service
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_email():
    """Test sending an email through Outlook"""
    try:
        logger.info("=" * 60)
        logger.info("Testing Outlook Email Service")
        logger.info("=" * 60)
        
        # Get email service
        logger.info("Step 1: Getting email service...")
        email_service = get_email_service()
        logger.info("✅ Email service initialized")
        
        # Test email
        test_email_address = input("Enter your email address to test: ").strip()
        
        if not test_email_address or '@' not in test_email_address:
            logger.error("❌ Invalid email address")
            return
        
        logger.info(f"Step 2: Sending test email to {test_email_address}...")
        
        success = email_service.send_email(
            to_email=test_email_address,
            subject="Test Email from Orbit Skill System",
            body="This is a test email to verify Outlook integration is working correctly.\n\nIf you receive this email, the system is working!",
            display_before_send=True  # Show email window for manual send
        )
        
        if success:
            logger.info("✅ Test email displayed in Outlook. Please check the Outlook window and send it manually.")
            logger.info("   After sending, check your Sent Items folder.")
        else:
            logger.error("❌ Failed to send test email")
            
    except Exception as e:
        logger.error(f"❌ Error during email test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_email()



