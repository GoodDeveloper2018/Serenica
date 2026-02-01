"""
WebSocket Connection Manager for real-time chat
"""

from typing import Dict, List
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, user_id: str, websocket: WebSocket):
        """Connect user"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, user_id: str):
        """Disconnect user"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected")
    
    async def send_personal_message(self, user_id: str, data: dict):
        """Send message to specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
                self.disconnect(user_id)
    
    async def broadcast(self, data: dict, exclude_user: str = None):
        """Broadcast to all connected users"""
        disconnected = []
        for user_id, connection in self.active_connections.items():
            if exclude_user and user_id == exclude_user:
                continue
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Failed to broadcast to {user_id}: {e}")
                disconnected.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected:
            self.disconnect(user_id)
    
    def get_active_connections_count(self) -> int:
        """Get count of active connections"""
        return len(self.active_connections)
