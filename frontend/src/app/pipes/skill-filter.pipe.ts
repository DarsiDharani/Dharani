/**
 * Skill Filter Pipe
 * 
 * Purpose: Filters skill arrays based on search text and status
 * Features:
 * - Case-insensitive search filtering by skill name
 * - Status-based filtering (e.g., 'Met', 'Gap')
 * - Handles null/undefined input gracefully
 * 
 * Usage in template:
 * {{ skills | skillFilter:searchText:statusFilter }}
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Pipe, PipeTransform } from '@angular/core';
import { Skill } from '../models/skill.model';

@Pipe({
  name: 'skillFilter'
})
export class SkillFilterPipe implements PipeTransform {
  /**
   * Transforms an array of skills by applying search and status filters
   * @param items - Array of Skill objects to filter (can be null)
   * @param search - Search text to match against skill names (case-insensitive)
   * @param status - Status value to filter by (e.g., 'Met', 'Gap')
   * @returns Filtered array of Skill objects
   */
  transform(items: Skill[] | null, search: string = '', status: string = ''): Skill[] {
    // Return empty array if input is null or undefined
    if (!items) return [];
    
    let filtered = items;
    
    // Apply search filter if search text is provided
    // Performs case-insensitive matching on skill name
    if (search) {
      filtered = filtered.filter(item => item.skill.toLowerCase().includes(search.toLowerCase()));
    }
    
    // Apply status filter if status is provided
    if (status) {
      filtered = filtered.filter(item => item.status === status);
    }
    
    return filtered;
  }
}
