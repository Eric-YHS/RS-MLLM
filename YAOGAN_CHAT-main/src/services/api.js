const API_BASE_URL = 'http://localhost:8000/api';

const getToken = () => localStorage.getItem('access_token');

const clearAuthData = () => {
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_info');
};

const authenticatedFetch = async (url, options = {}) => {
  const token = getToken();

  const headers = {
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, {
    ...options,
    headers
  });

  if (response.status === 401) {
    clearAuthData();
    window.dispatchEvent(new CustomEvent('authenticationFailed'));
    throw new Error('AUTHENTICATION_FAILED');
  }

  return response;
};

// eslint-disable-next-line no-unused-vars
const getUserInfo = () => {
  const userInfo = localStorage.getItem('user_info');
  return userInfo ? JSON.parse(userInfo) : null;
};

export const login = async (username, password) => {
  try {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await fetch(`${API_BASE_URL}/users/token`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Login failed');
    }

    const data = await response.json();

    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('user_info', JSON.stringify({
      id: data.user_id,
      username: data.username,
      displayName: data.display_name
    }));

    return data;
  } catch (error) {
    console.error('Error during login:', error);
    throw error;
  }
};

export const register = async (username, password, displayName) => {
  try {
    const response = await fetch(`${API_BASE_URL}/users/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        username,
        password,
        display_name: displayName
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Registration failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Error during registration:', error);
    throw error;
  }
};

export const getCurrentUser = async () => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await authenticatedFetch(`${API_BASE_URL}/users/me`);

    if (!response.ok) {
      throw new Error('Failed to get user information');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting user information:', error);
    throw error;
  }
};

export const getUserChats = async () => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await authenticatedFetch(`${API_BASE_URL}/users/chats`);

    if (!response.ok) {
      throw new Error('Failed to get chat history');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting chat history:', error);
    throw error;
  }
};

export const createNewChat = async (title = 'New Chat') => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await authenticatedFetch(`${API_BASE_URL}/users/chats?title=${encodeURIComponent(title)}`, {
      method: 'POST'
    });

    if (!response.ok) {
      throw new Error('Failed to create chat');
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating chat:', error);
    throw error;
  }
};

export const getChatMessages = async (chatId) => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await authenticatedFetch(`${API_BASE_URL}/users/chats/${chatId}/messages`);

    if (!response.ok) {
      throw new Error('Failed to get chat messages');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting chat messages:', error);
    throw error;
  }
};

export const deleteChat = async (chatId) => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await fetch(`${API_BASE_URL}/users/chats/${chatId}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to delete chat');
    }

    return await response.json();
  } catch (error) {
    console.error('Error deleting chat:', error);
    throw error;
  }
};

export const uploadAndAnalyzeImage = async (imageFile, prompt, taskType = 'description', chatId = null) => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('prompt', prompt);
    formData.append('task_type', taskType);
    if (chatId) {
      formData.append('chat_id', chatId);
    }

    const response = await fetch(`${API_BASE_URL}/analyze/image/`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to upload image');
    }

    return await response.json();
  } catch (error) {
    console.error('Error uploading and analyzing image:', error);
    throw error;
  }
};

export const getTaskResult = async (taskId) => {
  try {
    console.log(`Getting task result, task ID: ${taskId}`);
    const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/`);

    if (!response.ok) {
      const errorText = await response.text();
      let errorDetail;
      try {
        const errorData = JSON.parse(errorText);
        errorDetail = errorData.detail;
      } catch (e) {
        errorDetail = errorText;
      }
      throw new Error(errorDetail || `Failed to get task result, status code: ${response.status}`);
    }

    const data = await response.json();
    console.log(`Successfully got task, status: ${data.status}`);
    return data;
  } catch (error) {
    console.error('Error getting task result:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);

    if (!response.ok) {
      throw new Error(`Health check failed, status code: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error during health check:', error);
    throw error;
  }
};

export const processTextMessage = async (prompt, chatId, taskType = 'description') => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await fetch(`${API_BASE_URL}/chat/text`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt,
        chat_id: chatId,
        task_type: taskType
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process text message');
    }

    return await response.json();
  } catch (error) {
    console.error('Error processing text message:', error);
    throw error;
  }
};

export const processTextMessageAsync = async (prompt, chatId, taskType = 'description') => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await fetch(`${API_BASE_URL}/chat/text-async`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        prompt,
        chat_id: chatId,
        task_type: taskType
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to submit text message task');
    }

    return await response.json();
  } catch (error) {
    console.error('Error submitting text message task:', error);
    throw error;
  }
};

export const cancelTask = async (taskId) => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await fetch(`${API_BASE_URL}/cancel/${taskId}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      }
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to cancel task');
    }

    return await response.json();
  } catch (error) {
    console.error('Error canceling task:', error);
    throw error;
  }
};

export const updateChatTitle = async (chatId, title) => {
  try {
    const token = getToken();
    if (!token) {
      throw new Error('Not logged in');
    }

    const response = await fetch(`${API_BASE_URL}/users/chats/${chatId}/title`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ title })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to update chat title');
    }

    return await response.json();
  } catch (error) {
    console.error('Error updating chat title:', error);
    throw error;
  }
};

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export const pollTaskResult = async (
  taskId,
  interval = 3000,
  maxAttempts = 120,
  onProgress = null
) => {
  let attempts = 0;

  const poll = async () => {
    if (attempts >= maxAttempts) {
      throw new Error('Polling timed out, the task may still be processing');
    }

    attempts++;

    if (onProgress) {
      onProgress(attempts, maxAttempts);
    }

    try {
      const result = await getTaskResult(taskId);

      if (result.status === 'completed' || result.status === 'failed') {
        if (result.status === 'failed' && result.error) {
          throw new Error(`Task execution failed: ${result.error}`);
        }
        return result;
      }

      await delay(interval);

      return poll();
    } catch (error) {
      if (error.message && error.message.includes('Task not found')) {
        console.log(`Task ${taskId} is not ready yet, trying again later...`);
        await delay(interval);
        return poll();
      }
      throw error;
    }
  };

  return poll();
};