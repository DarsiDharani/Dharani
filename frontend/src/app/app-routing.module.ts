/**
 * Angular Application Routing Module
 * 
 * Purpose: Defines application routes and navigation configuration
 * Features:
 * - Configures all application routes
 * - Applies AuthGuard to protected routes (dashboards)
 * - Sets up default route redirection
 * 
 * Routes:
 * - /: Redirects to /home
 * - /home: Landing page
 * - /login: User login page
 * - /register: User registration page
 * - /about: About page
 * - /engineer-dashboard: Engineer dashboard (protected)
 * - /manager-dashboard: Manager dashboard (protected)
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AboutComponent } from './components/about/about.component';
import { HomeComponent } from './components/home/home.component';
import { LoginComponent } from './components/login/login.component';
import { RegisterComponent } from './components/register/register.component';

import { EngineerDashboardComponent } from './dashboards/engineer-dashboard/engineer-dashboard.component';
import { ManagerDashboardComponent } from './dashboards/manager-dashboard/manager-dashboard.component';

import { AuthGuard } from './guards/auth.guard';

/**
 * Application routes configuration
 * AuthGuard protects dashboard routes, requiring user authentication
 */
const routes: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: 'home', component: HomeComponent },
  { path: 'login', component: LoginComponent },
  { path: 'register', component: RegisterComponent },
  { path: 'about', component: AboutComponent },
  { path: 'engineer-dashboard', component: EngineerDashboardComponent, canActivate: [AuthGuard] },
  { path: 'manager-dashboard', component: ManagerDashboardComponent, canActivate: [AuthGuard] },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {}
