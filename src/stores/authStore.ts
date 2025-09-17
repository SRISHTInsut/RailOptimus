import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface User {
  id: string;
  name: string;
  section: string;
  username: string;
  email?: string;
  phone?: string;
  employeeId?: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  signup: (userData: Omit<User, 'id'> & { password: string }) => Promise<boolean>;
  logout: () => void;
  updateUser: (userData: Partial<User>) => void;
}

// Mock user database
const mockUsers: Array<User & { password: string }> = [
  {
    id: '1',
    name: 'John Controller',
    section: 'A1',
    username: 'controller1',
    password: 'password123'
  }
];

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,

      login: async (username: string, password: string) => {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const user = mockUsers.find(
          u => u.username === username && u.password === password
        );
        
        if (user) {
          const { password: _, ...userWithoutPassword } = user;
          set({ user: userWithoutPassword, isAuthenticated: true });
          return true;
        }
        
        return false;
      },

      signup: async (userData) => {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Check if username already exists
        const existingUser = mockUsers.find(u => u.username === userData.username);
        if (existingUser) {
          return false;
        }
        
        // Add new user
        const newUser = {
          ...userData,
          id: Date.now().toString(),
        };
        
        mockUsers.push(newUser);
        return true;
      },

      logout: () => {
        set({ user: null, isAuthenticated: false });
      },

      updateUser: (userData) => {
        const currentUser = get().user;
        if (currentUser) {
          const updatedUser = { ...currentUser, ...userData };
          set({ user: updatedUser });
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({ 
        user: state.user, 
        isAuthenticated: state.isAuthenticated 
      }),
    }
  )
);