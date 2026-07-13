"""
Authentication and Authorization Manager for Exam Hall Monitoring System
Handles user authentication, role-based access, and audit logging
"""

import json
import bcrypt
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

class AuthManager:
    def __init__(self, users_file='users.json', audit_log_file='logs/audit_log.json'):
        self.users_file = users_file
        self.audit_log_file = audit_log_file
        self.users = self.load_users()
        self.current_user = None
        
    def load_users(self):
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        else:
            # Create default admin user
            default_users = {
                "admin": {
                    "password": self.hash_password("admin123"),
                    "role": "admin",
                    "name": "Administrator",
                    "email": "admin@example.com",
                    "created_at": datetime.now().isoformat()
                },
                "proctor": {
                    "password": self.hash_password("proctor123"),
                    "role": "proctor",
                    "name": "Proctor User",
                    "email": "proctor@example.com",
                    "created_at": datetime.now().isoformat()
                }
            }
            self.save_users(default_users)
            return default_users
    
    def save_users(self, users=None):
        """Save users to JSON file"""
        users_to_save = users if users is not None else self.users
        with open(self.users_file, 'w') as f:
            json.dump(users_to_save, f, indent=2)
    
    def hash_password(self, password):
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password, hashed):
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def authenticate(self, username, password):
        """Authenticate user"""
        if username not in self.users:
            self.log_audit_event("login_failed", username, "User not found")
            return False, "Invalid username or password"
        
        user = self.users[username]
        if self.verify_password(password, user['password']):
            self.current_user = {
                'username': username,
                'role': user['role'],
                'name': user['name'],
                'email': user.get('email', '')
            }
            self.log_audit_event("login_success", username, "User logged in successfully")
            return True, "Login successful"
        else:
            self.log_audit_event("login_failed", username, "Invalid password")
            return False, "Invalid username or password"
    
    def logout(self):
        """Logout current user"""
        if self.current_user:
            self.log_audit_event("logout", self.current_user['username'], "User logged out")
            self.current_user = None
    
    def is_authenticated(self):
        """Check if user is authenticated"""
        return self.current_user is not None
    
    def has_role(self, role):
        """Check if current user has specific role"""
        if not self.is_authenticated():
            return False
        return self.current_user['role'] == role
    
    def is_admin(self):
        """Check if current user is admin"""
        return self.has_role('admin')
    
    def is_proctor(self):
        """Check if current user is proctor or admin"""
        return self.has_role('admin') or self.has_role('proctor')
    
    def add_user(self, username, password, role, name, email):
        """Add new user (admin only)"""
        if not self.is_admin():
            return False, "Only admins can add users"
        
        if username in self.users:
            return False, "Username already exists"
        
        self.users[username] = {
            "password": self.hash_password(password),
            "role": role,
            "name": name,
            "email": email,
            "created_at": datetime.now().isoformat()
        }
        self.save_users()
        self.log_audit_event("user_created", self.current_user['username'], f"Created user: {username}")
        return True, "User created successfully"
    
    def change_password(self, username, old_password, new_password):
        """Change user password"""
        if username not in self.users:
            return False, "User not found"
        
        # Only admin or the user themselves can change password
        if not self.is_admin() and self.current_user['username'] != username:
            return False, "Permission denied"
        
        user = self.users[username]
        if not self.verify_password(old_password, user['password']):
            return False, "Invalid old password"
        
        self.users[username]['password'] = self.hash_password(new_password)
        self.save_users()
        self.log_audit_event("password_changed", username, "Password changed successfully")
        return True, "Password changed successfully"
    
    def delete_user(self, username):
        """Delete user (admin only)"""
        if not self.is_admin():
            return False, "Only admins can delete users"
        
        if username == self.current_user['username']:
            return False, "Cannot delete yourself"
        
        if username not in self.users:
            return False, "User not found"
        
        del self.users[username]
        self.save_users()
        self.log_audit_event("user_deleted", self.current_user['username'], f"Deleted user: {username}")
        return True, "User deleted successfully"
    
    def get_all_users(self):
        """Get all users (admin only)"""
        if not self.is_admin():
            return []
        
        users_list = []
        for username, data in self.users.items():
            users_list.append({
                'username': username,
                'name': data['name'],
                'role': data['role'],
                'email': data.get('email', ''),
                'created_at': data.get('created_at', 'N/A')
            })
        return users_list
    
    def log_audit_event(self, event_type, username, description, metadata=None):
        """Log audit event ensuring logs directory exists"""
        log_dir = os.path.dirname(self.audit_log_file) or '.'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        audit_log = []
        if os.path.exists(self.audit_log_file):
            try:
                with open(self.audit_log_file, 'r') as f:
                    audit_log = json.load(f)
            except Exception:
                audit_log = []

        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'username': username,
            'description': description,
            'metadata': metadata or {}
        }
        audit_log.append(event)

        with open(self.audit_log_file, 'w') as f:
            json.dump(audit_log, f, indent=2)
    
    def get_audit_log(self, limit=100):
        """Get audit log entries (admin only)"""
        if not self.is_admin():
            return []
        
        if not os.path.exists(self.audit_log_file):
            return []
        
        with open(self.audit_log_file, 'r') as f:
            audit_log = json.load(f)
        
        # Return most recent entries
        return audit_log[-limit:][::-1]  # Reverse to show most recent first
    
    def get_audit_dataframe(self):
        """Get audit log as pandas DataFrame"""
        audit_log = self.get_audit_log(limit=1000)
        if not audit_log:
            return pd.DataFrame()
        
        return pd.DataFrame(audit_log)
    
    def clear_audit_log(self):
        """Clear audit log (admin only)"""
        if not self.is_admin():
            return False, "Only admins can clear audit log"
        
        if os.path.exists(self.audit_log_file):
            os.remove(self.audit_log_file)
        
        self.log_audit_event("audit_log_cleared", self.current_user['username'], "Audit log cleared")
        return True, "Audit log cleared successfully"
