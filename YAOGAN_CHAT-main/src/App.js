import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import Sidebar from './components/chat/Sidebar';
import ChatContainer from './components/chat/ChatContainer';
import CanvasDisplay from './components/chat/CanvasDisplay';
import { login, getUserChats, createNewChat } from './services/api';
import './components/chat/ChatLayout.css';

function App() {

  const [canvasSize] = useState({
    width: 800,
    height: 600
  });

  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [showRightSidebar, setShowRightSidebar] = useState(false);
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('access_token');
      const userInfo = localStorage.getItem('user_info');

      if (token && userInfo) {
        try {
          const { getCurrentUser } = require('./services/api');
          await getCurrentUser();

          const userData = JSON.parse(userInfo);
          setUser(userData);
          setIsAuthenticated(true);
        } catch (error) {
          console.error('Authentication check failed:', error);
          localStorage.removeItem('access_token');
          localStorage.removeItem('user_info');
          setIsAuthenticated(false);
        }
      } else {
        setIsAuthenticated(false);
      }
      setIsLoading(false);
    };

    checkAuth();

    const handleAuthFailure = () => {
      console.log('Authentication failure detected, redirecting to login page');
      setIsAuthenticated(false);
      setUser(null);
      setChats([]);
      setActiveChat(null);
    };

    window.addEventListener('authenticationFailed', handleAuthFailure);

    return () => {
      window.removeEventListener('authenticationFailed', handleAuthFailure);
    };
  }, []);

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      if (width < 1024) {
        setSidebarCollapsed(true);
      } else {
        setSidebarCollapsed(false);
      }

      if (width < 768 && showRightSidebar) {
        setShowRightSidebar(false);
      }
    };

    handleResize();

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [showRightSidebar]);

  const handleLogin = async (username, password) => {
    setIsLoggingIn(true);
    try {
      await login(username, password);
      const userInfo = JSON.parse(localStorage.getItem('user_info'));
      setUser(userInfo);
      setIsAuthenticated(true);
      await loadUserChats();
    } catch (error) {
      console.error('Login failed:', error);
      alert('Login failed: ' + error.message);
      throw error;
    } finally {
      setIsLoggingIn(false);
    }
  };

  const loadUserChats = async () => {
    try {
      const userChats = await getUserChats();
      if (userChats && userChats.length > 0) {
        const formattedChats = userChats.map(chat => ({
          id: chat.id,
          title: chat.title || 'New Chat',
          lastUpdated: new Date(chat.last_updated),
          lastMessage: chat.last_message
        }));
        setChats(formattedChats);
        setActiveChat(formattedChats[0]);
      } else {
        handleCreateNewChat();
      }
    } catch (error) {
      console.error('Failed to load chat history:', error);
      if (error.message === 'AUTHENTICATION_FAILED') {
        setIsAuthenticated(false);
        setUser(null);
        setChats([]);
        setActiveChat(null);
      }
    }
  };

  const handleCreateNewChat = useCallback(async () => {
    try {
      const newChat = await createNewChat('New Chat');
      const formattedChat = {
        id: newChat.id,
        title: newChat.title || 'New Chat',
        lastUpdated: new Date(newChat.last_updated)
      };

      setChats(prev => [formattedChat, ...prev]);
      setActiveChat(formattedChat);
    } catch (error) {
      console.error('Failed to create new chat:', error);
      if (error.message === 'AUTHENTICATION_FAILED') {
        setIsAuthenticated(false);
        setUser(null);
        setChats([]);
        setActiveChat(null);
      }
    }
  }, []);

  const handleDeleteChat = useCallback(async (chatId) => {
    try {
      const { deleteChat } = require('./services/api');
      await deleteChat(chatId);

      setChats(prev => prev.filter(chat => chat.id !== chatId));

      if (activeChat?.id === chatId) {
        const remainingChats = chats.filter(chat => chat.id !== chatId);
        if (remainingChats.length > 0) {
          setActiveChat(remainingChats[0]);
        } else {
          handleCreateNewChat();
        }
      }
    } catch (error) {
      console.error('Failed to delete chat:', error);
      alert('Failed to delete chat: ' + error.message);
    }
  }, [activeChat, chats, handleCreateNewChat]);

  useEffect(() => {
    if (isAuthenticated) {
      loadUserChats();
    }
  }, [isAuthenticated]);

  useEffect(() => {
    window.updateChatTitle = (chatId, newTitle) => {
      setChats(prevChats =>
        prevChats.map(chat =>
          chat.id === chatId
            ? { ...chat, title: newTitle }
            : chat
        )
      );
    };

    return () => {
      delete window.updateChatTitle;
    };
  }, []);

  const handleChatSelect = (chatId) => {
    const selected = chats.find(chat => chat.id === chatId);
    if (selected) {
      setActiveChat(selected);
    }
  };

  const handleFunctionSelect = (functionId) => {
    if (functionId === 'canvas') {
      setShowRightSidebar(prevState => !prevState);
    } else if (functionId === 'new-chat') {
      handleCreateNewChat();
    } else {
      console.log(`Selected function: ${functionId}`);
    }
  };

  if (isLoading) {
    return (
      <div className="loading">
        <div className="loading-spinner"></div>
        <div className="loading-text">Loading...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="login-container">
        <h1>YAOGAN</h1>
        <div className="login-card">
          <h2 className="login-title">Remote Sensing Image Analysis System</h2>
          <button
            onClick={() => handleLogin('demo', 'password')}
            className="login-button"
            disabled={isLoggingIn}
          >
            {isLoggingIn ? (
              <>
                <div className="loading-spinner" style={{ width: '20px', height: '20px', marginRight: '8px', display: 'inline-block' }}></div>
                Logging in...
              </>
            ) : (
              'Login with Demo Account'
            )}
          </button>
          <div className="login-info">
            <strong>Demo Account Info:</strong><br />
            Username: demo<br />
            Password: password
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      <div className="chat-layout">
        <div className={`sidebar-container ${sidebarCollapsed ? 'auto-collapsed' : ''}`}>
          <Sidebar
            chats={chats}
            activeChat={activeChat}
            onChatSelect={handleChatSelect}
            onNewChat={handleCreateNewChat}
            onDeleteChat={handleDeleteChat}
            user={user}
            setUser={setUser}
            setIsAuthenticated={setIsAuthenticated}
            setChats={setChats}
            setActiveChat={setActiveChat}
            forceCollapsed={sidebarCollapsed}
          />
        </div>

        <div className="main-content">
          <div className="chat-area">
            <ChatContainer
              activeChat={activeChat}
              onFunctionSelect={handleFunctionSelect}
            />
          </div>
          <div className={`right-sidebar ${showRightSidebar ? 'expanded' : ''}`}>
            <CanvasDisplay
              width={canvasSize.width * 0.8}
              height={canvasSize.height * 0.8}
              onClose={() => setShowRightSidebar(false)}
              activeChat={activeChat}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;