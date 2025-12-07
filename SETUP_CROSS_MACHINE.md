# Setting Up Skill Orbit Tool on Different Machines

This guide explains how to run the Skill Orbit Tool when the frontend and backend are on different machines (or when accessing from a different PC).

## Problem
When running on another PC, you may encounter:
- **401 Unauthorized** errors when trying to login
- CORS errors in the browser console
- Connection refused errors

## Solution

### Step 1: Find the Backend Machine's IP Address

On the machine running the backend server:

**Windows:**
```powershell
ipconfig
```
Look for "IPv4 Address" (usually something like `192.168.1.100` or `10.0.0.5`)

**Linux/Mac:**
```bash
ifconfig
# or
ip addr
```

### Step 2: Update Frontend Environment Configuration

1. Open `frontend/src/environments/environment.ts`
2. Change the `apiUrl` to point to the backend machine's IP address:

```typescript
export const environment = {
  production: false,
  apiUrl: 'http://192.168.1.100:8000'  // Replace with your backend machine's IP
};
```

**Example:**
- If backend is on `192.168.1.100`, use: `http://192.168.1.100:8000`
- If backend is on `10.0.0.5`, use: `http://10.0.0.5:8000`

### Step 3: Start the Backend Server

On the backend machine, make sure the server is running:

**Windows:**
```cmd
cd backend
start_server.bat
```

**Linux/Mac:**
```bash
cd backend
./start_server.sh
```

The server should start on `0.0.0.0:8000`, which means it will accept connections from any network interface.

### Step 4: Start the Frontend

On the frontend machine (or the same machine):

```bash
cd frontend
ng serve --host 0.0.0.0
```

The `--host 0.0.0.0` flag allows the Angular dev server to be accessed from other machines on the network.

### Step 5: Access the Application

Open your browser and navigate to:
- If frontend is on the same machine: `http://localhost:4200`
- If frontend is on another machine: `http://<frontend-machine-ip>:4200`

## Troubleshooting

### Still Getting 401 Unauthorized?

1. **Check if backend is running:**
   - Open browser and go to `http://<backend-ip>:8000/docs`
   - You should see the FastAPI documentation page

2. **Check firewall settings:**
   - Make sure port 8000 is open on the backend machine
   - Windows: Check Windows Firewall settings
   - Linux: Check iptables/ufw settings

3. **Verify credentials:**
   - Make sure you're using the correct username and password
   - Check if the user exists in the database

4. **Check network connectivity:**
   - Try pinging the backend machine: `ping <backend-ip>`
   - Try accessing the API directly: `http://<backend-ip>:8000/docs`

### CORS Errors?

The backend has been updated to allow all origins in development mode. If you still see CORS errors:

1. Make sure you've restarted the backend server after the updates
2. Clear your browser cache
3. Check the browser console for specific error messages

### Connection Refused?

1. Verify the backend server is running
2. Check that the IP address in `environment.ts` is correct
3. Ensure both machines are on the same network
4. Check firewall settings on the backend machine

## Production Setup

For production deployment, set the `CORS_ORIGINS` environment variable on the backend:

```bash
export CORS_ORIGINS="http://your-frontend-domain.com,https://your-frontend-domain.com"
```

Then update `frontend/src/environments/environment.prod.ts` (create it if it doesn't exist) with your production API URL.

