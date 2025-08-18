import React, { useState, useEffect, useRef } from 'react';
import './Sidebar.css';

const Sidebar = ({ chats, activeChat, onChatSelect, onNewChat, onDeleteChat, user, setUser, setIsAuthenticated, setChats, setActiveChat, forceCollapsed }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [isPinned, setIsPinned] = useState(true);
  const [isHovered, setIsHovered] = useState(false);
  const sidebarRef = useRef(null);

  const togglePin = (e) => {
    e.stopPropagation();
    setIsPinned(!isPinned);
    if (!isPinned) {
      setCollapsed(false);
    }
  };

  const handleMouseEnter = () => {
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
  };

  useEffect(() => {
    if (forceCollapsed) {
      setCollapsed(true);
    } else if (!isPinned) {
      setCollapsed(!isHovered);
    } else {
      setCollapsed(false);
    }
  }, [isHovered, isPinned, forceCollapsed]);

  return (
    <div
      ref={sidebarRef}
      className={`sidebar ${collapsed ? 'collapsed' : ''} ${isPinned ? 'pinned' : ''}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="app-title">
        <div className="app-name-bar">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            className="app-logo"
          >
            <path
              d="M12 2L2 7L12 12L22 7L12 2Z"
              stroke="black"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="rgba(255,255,255,0.2)"
            />
            <path
              d="M2 17L12 22L22 17"
              stroke="black"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <path
              d="M2 12L12 17L22 12"
              stroke="black"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <div className="app-name">YAOGAN</div>
        </div>
        <button
          className={`pin-sidebar ${isPinned ? 'active' : ''}`}
          onClick={togglePin}
          title={isPinned ? "Unpin sidebar" : "Pin sidebar"}
        >
          <div className="pin-icon"></div>
        </button>
      </div>

      <button className="new-chat-button" onClick={onNewChat}>
        <span className="button-icon"></span>
        <span className="button-text">New Chat</span>
      </button>

      <div className="chat-history">
        <h3 className="history-title">History</h3>
        <ul className="chat-list">
          {chats.map(chat => (
            <li
              key={chat.id}
              className={`chat-item ${activeChat?.id === chat.id ? 'active' : ''}`}
            >
              <div
                className="chat-item-content"
                onClick={() => onChatSelect(chat.id)}
              >
                <div className="chat-item-left">
                  <div className="chat-item-title gradient-text">{chat.title || 'New Chat'}</div>
                </div>
                <div className="chat-item-right">
                  <div className="chat-item-date">{formatDate(chat.lastUpdated)}</div>
                  <button
                    className="delete-chat-button"
                    title="Delete this chat"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm('Are you sure you want to delete this chat? This action cannot be undone.')) {
                        onDeleteChat && onDeleteChat(chat.id);
                      }
                    }}
                  >
                    <span className="delete-icon">×</span>
                  </button>
                </div>
              </div>
            </li>
          ))}

          {chats.length === 0 && (
            <li className="no-chats-message">No chat history</li>
          )}
        </ul>
      </div>

      <div className="user-profile">
        <div className="user-profile-inner">
          <div className="user-avatar">
            {user?.displayName?.charAt(0) || user?.username?.charAt(0) || '?'}
          </div>
          <div className="user-info">
            <div className="user-name">{user?.displayName || user?.username || 'Not logged in'}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

const formatDate = (timestamp) => {
  if (!timestamp) return '';

  const date = new Date(timestamp);
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);

  const isToday = date.toDateString() === today.toDateString();
  const isYesterday = date.toDateString() === yesterday.toDateString();

  if (isToday) {
    return 'Today';
  } else if (isYesterday) {
    return 'Yesterday';
  } else {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric'
    }).format(date);
  }
};

export default Sidebar;