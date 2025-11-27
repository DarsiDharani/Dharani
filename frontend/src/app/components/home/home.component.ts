/**
 * Home Component
 * 
 * Purpose: Landing page with carousel/slideshow showcasing application features
 * Features:
 * - Auto-rotating image carousel
 * - Manual navigation (next/prev buttons)
 * - Direct slide selection
 * - 3D rotation effects
 * - Automatic cleanup on component destruction
 * 
 * @author Orbit Skill Development Team
 * @date 2025
 */

import { Component, OnInit, OnDestroy } from '@angular/core';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit, OnDestroy {
  /** Array of slide objects containing image URLs, titles, and descriptions */
  slides = [
    {
      url: '/assets/Home8.png',
      title: 'A Unified Skill Ecosystem',
      description: 'Empowering organizations to manage, track, and grow employee skills effectively.',
    },
    {
      url: '/assets/Home5.png',
      title: 'Data-Driven Growth Paths',
      description: 'Manage skill levels across multiple roles and departments with precision.',
    },
    {
      url: '/assets/Home6.png',
      title: 'Intelligent Analytics',
      description: 'Track progress with intelligent dashboards and real-time feedback.',
    },
    {
      url: '/assets/Home2.png',
      title: 'Dynamic Skill Assessments',
      description: 'Assess employee skills with dynamic and adaptive evaluation tools.',
    }
  ];

  /** Current active slide index (0-based) */
  currentIndex = 0;
  
  /** Rotation angle for 3D carousel effect (in degrees) */
  rotationAngle = 0;
  
  /** Reference to the auto-slide interval timer */
  slideInterval: any;
  
  /** Duration between automatic slide transitions (in milliseconds) */
  slideDuration = 2000;

  /**
   * Angular lifecycle hook - initializes auto-slide on component load
   */
  ngOnInit() {
    this.startAutoSlide();
  }

  /**
   * Angular lifecycle hook - cleans up interval timer to prevent memory leaks
   */
  ngOnDestroy() {
    this.stopAutoSlide();
  }

  /**
   * Starts automatic slide rotation
   * Stops any existing interval before starting a new one to prevent duplicates
   */
  startAutoSlide() {
    this.stopAutoSlide();
    this.slideInterval = setInterval(() => {
      this.nextSlide();
    }, this.slideDuration);
  }

  /**
   * Stops automatic slide rotation
   * Clears the interval timer to prevent memory leaks
   */
  stopAutoSlide() {
    clearInterval(this.slideInterval);
  }

  /**
   * Updates the 3D rotation angle based on current slide index
   * For 4 slides, each slide represents 90 degrees of rotation
   * Resets auto-slide timer when called (user interaction detected)
   */
  private updateCarouselRotation() {
    // For 4 slides, each turn is 90 degrees
    this.rotationAngle = this.currentIndex * -90;
    // Reset timer on manual interaction to give user time to view slide
    this.startAutoSlide();
  }

  /**
   * Advances to the next slide
   * Wraps around to first slide when reaching the end
   */
  nextSlide() {
    this.currentIndex = (this.currentIndex + 1) % this.slides.length;
    this.updateCarouselRotation();
  }

  /**
   * Goes back to the previous slide
   * Wraps around to last slide when at the beginning
   */
  prevSlide() {
    this.currentIndex = (this.currentIndex - 1 + this.slides.length) % this.slides.length;
    this.updateCarouselRotation();
  }

  /**
   * Jumps directly to a specific slide by index
   * @param index - Zero-based index of the slide to display
   */
  goToSlide(index: number) {
    this.currentIndex = index;
    this.updateCarouselRotation();
  }
}