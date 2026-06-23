/**
 * Disease Prediction and Data Analysis System - Frontend Performance Optimized Version
 * Includes efficient animations, lazy loading, and page performance optimizations
 */

// Configuration
const CONFIG = {
  // Basic Configuration
  SCROLL_THRESHOLD: 200,        // Scroll trigger threshold
  SCROLL_DEBOUNCE: 20,          // Scroll debounce time (milliseconds)
  ANIMATION_THRESHOLD: 0.2,     // Animation trigger threshold (element ratio in viewport)
  IS_MOBILE: window.innerWidth < 768, // Whether it's a mobile device
  
  // Performance Settings
  ENABLE_ANIMATIONS: true,      // Global animation toggle
  LAZY_LOAD_IMAGES: true,       // Image lazy loading
  ENABLE_TRANSITIONS: true,     // Enable page transitions
  
  // CSS Class Names
  CLASSES: {
    visible: 'is-visible',
    fadeInSection: 'fade-in-section',
    scrollToTop: 'scroll-to-top',
    scrollIndicator: 'scroll-indicator'
  }
};

/**
 * Utility Functions
 */
const Utils = {
  // Debounce function - reduce high-frequency event triggers
  debounce(func, wait) {
    let timeout;
    return function(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  },
  
  // Check if element is within viewport
  isElementInViewport(el, threshold = CONFIG.ANIMATION_THRESHOLD) {
    if (!el) return false;
    
    const rect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight || document.documentElement.clientHeight;
    
    // Element is visible when top enters a certain percentage of viewport bottom
    return (
      rect.top <= windowHeight * (1 - threshold) &&
      rect.bottom >= 0
    );
  },
  
  // Detect device performance
  checkDevicePerformance() {
    // Detect mobile devices
    if (/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)) {
      CONFIG.IS_MOBILE = true;
    }
    
    // If device is low-end mobile, disable some complex animations
    if (CONFIG.IS_MOBILE && navigator.hardwareConcurrency && navigator.hardwareConcurrency < 4) {
      CONFIG.ENABLE_ANIMATIONS = false;
    }
  },
  
  // Add one-time event listener
  once(element, eventType, callback) {
    const handler = (e) => {
      callback(e);
      element.removeEventListener(eventType, handler);
    };
    element.addEventListener(eventType, handler);
  }
};

/**
 * Animation Controller
 */
const AnimationController = {
  // Initialize visibility observer
  initIntersectionObserver() {
    if (!('IntersectionObserver' in window) || !CONFIG.ENABLE_ANIMATIONS) {
      // If IntersectionObserver is not supported or animations are disabled, make all elements visible immediately
      document.querySelectorAll(`.${CONFIG.CLASSES.fadeInSection}`).forEach(el => {
        el.classList.add(CONFIG.CLASSES.visible);
      });
      return;
    }
    
    const options = {
      root: null,
      rootMargin: '0px',
      threshold: 0.15
    };
    
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add(CONFIG.CLASSES.visible);
          observer.unobserve(entry.target); // Stop observing after element is visible
        }
      });
    }, options);
    
    // Observe all elements that need animation
    document.querySelectorAll(`.${CONFIG.CLASSES.fadeInSection}`).forEach(el => {
      observer.observe(el);
    });
  },
  
  // Prepare element animations
  prepareElements() {
    // Convert all content sections to animatable elements
    const sections = document.querySelectorAll('.jumbotron, .row, .card-container, .chart-container, .content-block, .slide-section');
    
    sections.forEach(section => {
      if (!section.classList.contains(CONFIG.CLASSES.fadeInSection)) {
        section.classList.add(CONFIG.CLASSES.fadeInSection);
      }
    });
    
    // If animations are disabled, show all elements immediately
    if (!CONFIG.ENABLE_ANIMATIONS) {
      sections.forEach(section => {
        section.classList.add(CONFIG.CLASSES.visible);
      });
    }
  }
};

/**
 * Page Scroll Control
 */
const ScrollController = {
  // Scroll indicator
  initScrollIndicator() {
    const scrollIndicator = document.querySelector(`.${CONFIG.CLASSES.scrollIndicator}`) || 
                           this.createScrollIndicator();
    
    this.updateScrollIndicator = () => {
      const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (window.scrollY / windowHeight) * 100;
      
      if (scrollIndicator) {
        scrollIndicator.style.width = scrolled + '%';
      }
    };
  },
  
  // Create scroll indicator element
  createScrollIndicator() {
    const indicator = document.createElement('div');
    indicator.className = CONFIG.CLASSES.scrollIndicator;
    document.body.appendChild(indicator);
    return indicator;
  },
  
  // Initialize back to top button
  initScrollToTopButton() {
    const scrollToTopBtn = document.querySelector(`.${CONFIG.CLASSES.scrollToTop}`) || 
                          this.createScrollToTopButton();
    
    // Update button visibility
    this.updateScrollToTopButton = () => {
      if (window.scrollY > CONFIG.SCROLL_THRESHOLD) {
        scrollToTopBtn.classList.add('visible');
      } else {
        scrollToTopBtn.classList.remove('visible');
      }
    };
  },
  
  // Create back to top button
  createScrollToTopButton() {
    const button = document.createElement('div');
    button.className = CONFIG.CLASSES.scrollToTop;
    button.innerHTML = '<i class="fas fa-arrow-up"></i>';
    button.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    document.body.appendChild(button);
    return button;
  },
  
  // Handle scroll events
  handleScroll() {
    // Update scroll indicator
    this.updateScrollIndicator();
    
    // Update back to top button
    this.updateScrollToTopButton();
  },
  
  // Initialize page scroll handling
  init() {
    this.initScrollIndicator();
    this.initScrollToTopButton();
    
    // Use debounce to optimize scroll event handling
    const debouncedScrollHandler = Utils.debounce(
      () => this.handleScroll(), 
      CONFIG.SCROLL_DEBOUNCE
    );
    
    window.addEventListener('scroll', debouncedScrollHandler, { passive: true });
    
    // Trigger scroll handling once initially
    this.handleScroll();
  }
};

/**
 * Image Lazy Load Controller
 */
const LazyLoadController = {
  init() {
    if (!CONFIG.LAZY_LOAD_IMAGES) {
      // If lazy loading is disabled, load all images directly
      document.querySelectorAll('img[data-src]').forEach(img => {
        img.src = img.getAttribute('data-src');
        img.classList.add('loaded');
      });
      return;
    }
    
    if ('IntersectionObserver' in window) {
      const imgObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            const src = img.getAttribute('data-src');
            
            if (src) {
              // Create a temporary image to preload
              const tempImage = new Image();
              tempImage.onload = () => {
                img.src = src;
                img.classList.add('loaded');
              };
              tempImage.src = src;
              
              // Remove data-src attribute and stop observing
              img.removeAttribute('data-src');
              observer.unobserve(img);
            }
          }
        });
      }, {
        rootMargin: '50px'  // Start loading 50px early
      });

      // Observe all lazy-loaded images
      document.querySelectorAll('img[data-src]').forEach(img => {
        img.classList.add('lazy');
        imgObserver.observe(img);
      });
    } else {
      // Fallback: when IntersectionObserver is not supported
      document.querySelectorAll('img[data-src]').forEach(img => {
        img.src = img.getAttribute('data-src');
        img.classList.add('loaded');
      });
    }
  }
};

/**
 * Page Transition Controller
 */
const PageTransitionController = {
  init() {
    if (!CONFIG.ENABLE_TRANSITIONS) return;
    
    // Create page transition element
    const transitionElement = document.createElement('div');
    transitionElement.className = 'page-transition';
    document.body.appendChild(transitionElement);
    
    // Handle internal link clicks
    document.querySelectorAll('a').forEach(link => {
      // Only handle same-site links, exclude dropdown items
      if (link.hostname === window.location.hostname && 
          !link.hasAttribute('data-no-transition') && 
          !link.classList.contains('dropdown-item')) {
        
        link.addEventListener('click', (e) => {
          // For anchor links within the same page, use smooth scroll
          if (link.hash && document.querySelector(link.hash)) {
            e.preventDefault();
            const targetElement = document.querySelector(link.hash);
            
            // Smooth scroll to target element
            window.scrollTo({
              top: targetElement.offsetTop - 70,
              behavior: 'smooth'
            });
          } 
          // For links to other pages on the site, add transition effect
          else if (!link.hash && link.pathname !== window.location.pathname) {
            e.preventDefault();
            
            // Activate transition effect
            transitionElement.classList.add('active');
            
            // Navigate to target page after 300ms
            setTimeout(() => {
              window.location.href = link.href;
            }, 300);
          }
        });
      }
    });
    
    // Fade in effect after page load
    document.body.classList.add('fade-in');
  }
};

/**
 * Tab Controller
 */
const TabController = {
  init() {
    const tabLinks = document.querySelectorAll('[data-bs-toggle="tab"]');
    
    tabLinks.forEach(tabLink => {
      tabLink.addEventListener('shown.bs.tab', (e) => {
        const targetTabPane = document.querySelector(e.target.getAttribute('data-bs-target'));
        
        if (targetTabPane) {
          // Ensure content enters viewport
          setTimeout(() => {
            // Use more efficient scrollIntoView method
            targetTabPane.scrollIntoView({
              behavior: 'smooth',
              block: 'nearest'
            });
          }, 50);
          
          // Activate charts and images in current tab
          this.activateTabContent(targetTabPane);
        }
      });
    });
  },
  
  // Activate content in tab
  activateTabContent(container) {
    // Load lazy-loaded images
    container.querySelectorAll('img[data-src]').forEach(img => {
      if (img.getAttribute('data-src')) {
        img.src = img.getAttribute('data-src');
        img.classList.add('loaded');
        img.removeAttribute('data-src');
      }
    });
    
    // Make all animated elements visible
    container.querySelectorAll(`.${CONFIG.CLASSES.fadeInSection}`).forEach(el => {
      el.classList.add(CONFIG.CLASSES.visible);
    });
  }
};

/**
 * Background Controller
 */
const BackgroundController = {
  init() {
    const backgrounds = document.querySelectorAll('.parallax-bg');
    if (backgrounds.length === 0) return;
    
    let currentBg = 0;
    backgrounds[0].classList.add('active');
    
    // Use debounce for performance optimization
    const debouncedBackgroundHandler = Utils.debounce(() => {
      const scrollPosition = window.scrollY;
      const windowHeight = window.innerHeight;
      const documentHeight = document.body.scrollHeight;
      
      // Calculate which background image should be displayed based on scroll position
      const scrollRatio = Math.min(scrollPosition / (documentHeight - windowHeight), 1);
      const bgIndex = Math.min(Math.floor(scrollRatio * backgrounds.length), backgrounds.length - 1);
      
      // If background image needs to change
      if (bgIndex !== currentBg) {
        // Remove current active background
        backgrounds[currentBg].classList.remove('active');
        // Activate new background
        backgrounds[bgIndex].classList.add('active');
        // Update current background index
        currentBg = bgIndex;
      }
    }, 100);
    
    window.addEventListener('scroll', debouncedBackgroundHandler, { passive: true });
  }
};

/**
 * Prediction Form Controller
 */
const PredictionFormController = {
  init() {
    // Check if on prediction page
    const predictionForm = document.querySelector('#prediction-form');
    if (!predictionForm) return;
    
    // Add submit event listener to prediction form
    predictionForm.addEventListener('submit', (e) => {
      e.preventDefault();
      
      // Show loading animation
      const submitBtn = predictionForm.querySelector('button[type="submit"]');
      const originalBtnText = submitBtn.innerHTML;
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';
      
      // Get form data
      const formData = new FormData(predictionForm);
      
      // Send AJAX request
      fetch('/predict', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Server response error: ' + response.status);
        }
        return response.json();
      })
      .then(data => {
        // Restore submit button
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
        
        // Show prediction results
        if (data.error) {
          this.showPredictionError(new Error(data.error));
        } else {
          this.showPredictionResult(data);
        }
      })
      .catch(error => {
        // Restore submit button
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalBtnText;
        
        // Show error
        this.showPredictionError(error);
      });
    });
    
    // Add disease type switch event
    const diseaseTypeSelect = document.querySelector('#disease_type');
    if (diseaseTypeSelect) {
      diseaseTypeSelect.addEventListener('change', () => {
        // Show corresponding form fields
        this.toggleFormFields(diseaseTypeSelect.value);
      });
      
      // Initialize form field display
      this.toggleFormFields(diseaseTypeSelect.value);
    }
  },
  
  // Toggle form fields
  toggleFormFields(diseaseType) {
    // Get all form field groups
    const fieldGroups = document.querySelectorAll('.form-field-group');
    
    // Hide all field groups
    fieldGroups.forEach(group => {
      group.style.display = 'none';
    });
    
    // Show field groups for corresponding disease type
    const targetGroup = document.querySelector(`.form-field-group[data-disease="${diseaseType}"]`);
    if (targetGroup) {
      targetGroup.style.display = 'block';
      
      // Add fade-in effect
      setTimeout(() => {
        targetGroup.classList.add('fade-in');
      }, 50);
    }
  },
  
  // Show prediction results
  showPredictionResult(data) {
    // Get result container
    const resultContainer = document.querySelector('#prediction-result');
    if (!resultContainer) return;
    
    // Clear existing content
    resultContainer.innerHTML = '';
    resultContainer.style.display = 'block';
    
    // Create result card
    const resultCard = document.createElement('div');
    resultCard.className = 'card fade-in';
    
    // Create corresponding result card content based on disease type
    if (data.disease_type === 'stroke') {
      this.createStrokeResultCard(resultCard, data);
    } else if (data.disease_type === 'heart') {
      this.createHeartResultCard(resultCard, data);
    } else if (data.disease_type === 'cirrhosis') {
      this.createCirrhosisResultCard(resultCard, data);
    }
    
    // If mock data was used, add notice
    if (data.note && data.note.includes('Mock data')) {
      const noteAlert = document.createElement('div');
      noteAlert.className = 'alert alert-info mt-3';
      noteAlert.innerHTML = `
        <small><i class="fas fa-info-circle"></i> Note: ${data.note}. Actual results may vary.</small>
      `;
      resultCard.appendChild(noteAlert);
    }
    
    // Add to result container
    resultContainer.appendChild(resultCard);
    
    // Scroll to result area
    resultContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
    
    // Initialize risk gauge
    this.initRiskGauge();
  },
  
  // Create stroke result card
  createStrokeResultCard(card, data) {
    const isPredicted = data.prediction === 1;
    const probability = data.probability["1"] * 100;
    
    // Check if raw probability data exists (before calibration)
    let calibrationInfo = '';
    if (data.raw_probability && data.calibrated) {
      const rawProbability = data.raw_probability["1"] * 100;
      calibrationInfo = `
        <div class="alert alert-info mt-3">
          <h5><i class="fas fa-info-circle"></i> Model Calibration Information</h5>
          <p>Raw predicted risk: <strong>${rawProbability.toFixed(2)}%</strong> → Calibrated risk: <strong>${probability.toFixed(2)}%</strong></p>
          <small>We applied advanced probability calibration techniques, specifically optimized for high-risk groups to make predictions more aligned with actual clinical outcomes.</small>
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="card-header bg-${isPredicted ? 'danger' : 'success'} text-white">
        <h3 class="card-title mb-0">Stroke Risk Prediction Results</h3>
      </div>
      <div class="card-body">
        <div class="text-center mb-4">
          <div class="risk-gauge" data-risk="${probability}">
            <div class="gauge-value">${probability.toFixed(2)}%</div>
          </div>
        </div>
        <h4 class="text-center mb-3">Risk Assessment: <span class="${isPredicted ? 'text-danger' : 'text-success'}">${isPredicted ? 'High Risk' : 'Low Risk'}</span></h4>
        <div class="progress mb-4">
          <div class="progress-bar bg-${this.getProgressBarColor(probability)}" role="progressbar" style="width: ${probability}%" aria-valuenow="${probability}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        ${calibrationInfo}
        <div class="alert alert-${isPredicted ? 'warning' : 'info'}">
          <p>${isPredicted ? 
            'Based on the data you provided, the model predicts you have a higher stroke risk. Please consider consulting a doctor for a more detailed assessment.' : 
            'Based on the data you provided, the model predicts you currently have a lower stroke risk. Keep up the healthy lifestyle!'}
          </p>
        </div>
        <div class="mt-4">
          <h5>Prevention Tips:</h5>
          <ul>
            <li>Regularly monitor blood pressure and keep it within healthy range</li>
            <li>Maintain a healthy diet, reduce sodium and saturated fat intake</li>
            <li>Engage in at least 150 minutes of moderate-intensity aerobic exercise per week</li>
            <li>Quit smoking and limit alcohol</li>
            <li>Have regular health checkups, especially focusing on cardiovascular health</li>
          </ul>
        </div>
      </div>
    `;
  },
  
  // Create heart disease result card
  createHeartResultCard(card, data) {
    const isPredicted = data.prediction === 1;
    const probability = data.probability["1"] * 100;
    
    // Check if raw probability data exists (before calibration)
    let calibrationInfo = '';
    if (data.raw_probability && data.calibrated) {
      const rawProbability = data.raw_probability["1"] * 100;
      calibrationInfo = `
        <div class="alert alert-info mt-3">
          <h5><i class="fas fa-info-circle"></i> Model Calibration Information</h5>
          <p>Raw predicted risk: <strong>${rawProbability.toFixed(2)}%</strong> → Calibrated risk: <strong>${probability.toFixed(2)}%</strong></p>
          <small>We applied advanced probability calibration techniques, specifically optimized for high-risk groups to make predictions more aligned with actual clinical outcomes.</small>
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="card-header bg-${isPredicted ? 'danger' : 'success'} text-white">
        <h3 class="card-title mb-0">Heart Disease Risk Prediction Results</h3>
      </div>
      <div class="card-body">
        <div class="text-center mb-4">
          <div class="risk-gauge" data-risk="${probability}">
            <div class="gauge-value">${probability.toFixed(2)}%</div>
          </div>
        </div>
        <h4 class="text-center mb-3">Risk Assessment: <span class="${isPredicted ? 'text-danger' : 'text-success'}">${isPredicted ? 'High Risk' : 'Low Risk'}</span></h4>
        <div class="progress mb-4">
          <div class="progress-bar bg-${this.getProgressBarColor(probability)}" role="progressbar" style="width: ${probability}%" aria-valuenow="${probability}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        ${calibrationInfo}
        <div class="alert alert-${isPredicted ? 'warning' : 'info'}">
          <p>${isPredicted ? 
            'Based on the data you provided, the model predicts you have a higher heart disease risk. We recommend consulting a cardiologist for further evaluation.' : 
            'Based on the data you provided, the model predicts you currently have a lower heart disease risk. Keep up the healthy lifestyle!'}
          </p>
        </div>
        <div class="mt-4">
          <h5>Prevention Tips:</h5>
          <ul>
            <li>Control blood pressure and cholesterol levels</li>
            <li>Engage in moderate exercise daily</li>
            <li>Maintain a balanced diet with more fruits and vegetables</li>
            <li>Keep a healthy weight</li>
            <li>Manage stress and ensure adequate sleep</li>
          </ul>
        </div>
      </div>
    `;
  },
  
  // Create cirrhosis result card
  createCirrhosisResultCard(card, data) {
    const stagePrediction = Math.round(data.prediction);
    const stageDescriptions = {
      1: 'Early-stage cirrhosis, liver still functioning normally',
      2: 'Moderate cirrhosis, mild liver function impairment',
      3: 'Advanced cirrhosis, moderate liver function impairment',
      4: 'Late-stage cirrhosis, severe liver function impairment'
    };
    
    const stageRiskClass = {
      1: 'success',
      2: 'info',
      3: 'warning',
      4: 'danger'
    };
    
    // Check if raw prediction data exists (before calibration)
    let calibrationInfo = '';
    if (data.raw_prediction && data.calibrated) {
      const rawStagePrediction = Math.round(data.raw_prediction);
      calibrationInfo = `
        <div class="alert alert-info mt-3">
          <h5><i class="fas fa-info-circle"></i> Model Calibration Information</h5>
          <p>Raw predicted stage: <strong>Stage ${rawStagePrediction}</strong> → Calibrated stage: <strong>Stage ${stagePrediction}</strong></p>
          <small>We applied advanced probability calibration techniques, specifically optimized for high-risk groups to make predictions more aligned with actual clinical outcomes.</small>
        </div>
      `;
    }
    
    card.innerHTML = `
      <div class="card-header bg-${stageRiskClass[stagePrediction]} text-${stagePrediction === 4 ? 'white' : 'dark'}">
        <h3 class="card-title mb-0">Cirrhosis Stage Prediction Results</h3>
      </div>
      <div class="card-body">
        <div class="text-center mb-4">
          <div class="stage-indicator">
            <span class="stage-number stage-${stagePrediction}">${stagePrediction}</span>
            <span class="stage-label">Stage</span>
          </div>
        </div>
        <h4 class="text-center mb-3">Stage Assessment: <span class="text-${stageRiskClass[stagePrediction]}">${stageDescriptions[stagePrediction]}</span></h4>
        <div class="progress mb-4">
          <div class="progress-bar bg-${stageRiskClass[stagePrediction]}" role="progressbar" style="width: ${stagePrediction * 25}%" aria-valuenow="${stagePrediction * 25}" aria-valuemin="0" aria-valuemax="100"></div>
        </div>
        ${calibrationInfo}
        <div class="alert alert-${stageRiskClass[stagePrediction]}">
          <p>Based on the data you provided, the model predicts your cirrhosis is at Stage ${stagePrediction}. ${
            stagePrediction <= 2 ? 
            'Early detection is important, we recommend regular follow-ups and following medical advice.' : 
            'Please be sure to follow the treatment recommendations of your specialist doctor and have regular liver function tests.'
          }</p>
        </div>
        <div class="mt-4">
          <h5>Health Management Tips:</h5>
          <ul>
            <li>Strictly follow prescribed medications as directed by your doctor</li>
            <li>Maintain a healthy diet and limit salt intake</li>
            <li>Avoid alcohol and other hepatotoxic substances</li>
            <li>Regularly monitor liver function</li>
            <li>Get adequate rest and avoid overexertion</li>
          </ul>
        </div>
      </div>
    `;
  },
  
  // Show prediction error
  showPredictionError(error) {
    const resultContainer = document.querySelector('#prediction-result');
    if (!resultContainer) return;
    
    resultContainer.innerHTML = '';
    resultContainer.style.display = 'block';
    
    const errorAlert = document.createElement('div');
    errorAlert.className = 'alert alert-danger fade-in';
    
    // Check if it's a model not loaded error
    const isModelError = error.message && 
      (error.message.includes('Model not loaded') || 
       error.message.includes('model') || 
       error.message.includes('Failed to load'));
    
    if (isModelError) {
      errorAlert.innerHTML = `
        <h4 class="alert-heading"><i class="fas fa-exclamation-triangle"></i> Error Occurred</h4>
        <p>We apologize, an issue occurred during prediction: Model not loaded, please try again later</p>
        <hr>
        <p class="mb-0">Please check your input data and try again.</p>
      `;
    } else {
      errorAlert.innerHTML = `
        <h4 class="alert-heading"><i class="fas fa-exclamation-triangle"></i> Prediction Error</h4>
        <p>We apologize, an error occurred during prediction. Please check input data and try again.</p>
        <hr>
        <p class="mb-0">Error details: ${error.message || 'Unknown error'}</p>
      `;
    }
    
    resultContainer.appendChild(errorAlert);
    resultContainer.scrollIntoView({behavior: 'smooth', block: 'nearest'});
  },
  
  // Initialize risk gauge
  initRiskGauge() {
    const gauges = document.querySelectorAll('.risk-gauge');
    
    gauges.forEach(gauge => {
      const risk = parseFloat(gauge.dataset.risk);
      const angle = (risk / 100) * 180 - 90; // Convert to angle, -90 to 90 degrees
      
      // Set gauge style - use CSS variables to avoid repaints
      gauge.style.setProperty('--risk-angle', `${angle}deg`);
      gauge.style.background = `conic-gradient(
        ${this.getRiskColor(risk)} ${angle}deg,
        #e0e0e0 ${angle}deg
      )`;
    });
  },
  
  // Get progress bar color based on risk value
  getProgressBarColor(riskValue) {
    if (riskValue < 20) return 'success';
    if (riskValue < 50) return 'info';
    if (riskValue < 80) return 'warning';
    return 'danger';
  },
  
  // Get color based on risk value
  getRiskColor(risk) {
    if (risk < 20) return '#28a745'; // Green
    if (risk < 50) return '#17a2b8'; // Blue
    if (risk < 80) return '#ffc107'; // Yellow
    return '#dc3545'; // Red
  }
};

/**
 * Navigation Menu Controller
 */
const NavController = {
  init() {
    // Initialize mobile navigation
    this.setupMobileNav();
  },
  
  setupMobileNav() {
    const navbarToggler = document.querySelector('.navbar-toggler');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggler && navbarCollapse) {
      navbarToggler.addEventListener('click', () => {
        navbarCollapse.classList.toggle('show');
      });
      
      // Auto-close mobile navigation menu after clicking nav items
      const navLinks = navbarCollapse.querySelectorAll('.nav-link');
      navLinks.forEach(link => {
        link.addEventListener('click', () => {
          navbarCollapse.classList.remove('show');
        });
      });
    }
  }
};

/**
 * Main Application Initialization
 */
document.addEventListener('DOMContentLoaded', () => {
  try {
    // Detect device performance and decide whether to enable animations
    Utils.checkDevicePerformance();
    
    // Initialize tooltips and popovers
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if(tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Initialize background controller
    if(typeof BackgroundController !== 'undefined') {
      BackgroundController.init();
    }
    
    // Prepare animation elements
    if(typeof AnimationController !== 'undefined') {
      AnimationController.prepareElements();
      AnimationController.initIntersectionObserver();
    }
    
    // Initialize scroll control
    if(typeof ScrollController !== 'undefined') {
      ScrollController.init();
    }
    
    // Initialize image lazy loading
    if(typeof LazyLoadController !== 'undefined') {
      LazyLoadController.init();
    }
    
    // Initialize page transition effects
    if(typeof PageTransitionController !== 'undefined') {
      PageTransitionController.init();
    }
    
    // Initialize tab control
    if(typeof TabController !== 'undefined') {
      TabController.init();
    }
    
    // Initialize prediction form control
    if(typeof PredictionFormController !== 'undefined') {
      PredictionFormController.init();
    }
    
    // Initialize navigation controller
    if(typeof NavController !== 'undefined') {
      NavController.init();
    }
    
    // Initialize chart controller
    if(typeof ChartController !== 'undefined') {
      ChartController.init();
    }
    
    // Trigger scroll event once to ensure animations display correctly on initial load
    window.dispatchEvent(new Event('scroll'));
    
    console.log('All system components initialized');
  } catch(e) {
    console.error('Initialization failed: ', e);
  }
});

/* ============================================================
   Additional Performance Optimizations: Skeleton Screen, Prediction Cache, Request Deduplication
   ============================================================ */

// Prediction result cache (avoid duplicate requests)
const PredictionCache = {
  cache: new Map(),
  maxAge: 5 * 60 * 1000, // 5 minute cache

  makeKey(formData) {
    const entries = [...formData.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    return entries.map(([k, v]) => `${k}=${v}`).join('&');
  },

  get(key) {
    const entry = this.cache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.ts > this.maxAge) {
      this.cache.delete(key);
      return null;
    }
    return entry.data;
  },

  set(key, data) {
    this.cache.set(key, { data, ts: Date.now() });
    if (this.cache.size > 50) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
  }
};

// Request deduplication (prevent rapid repeated clicks from users)
const RequestDedupe = {
  pending: new Map(),

  check(key) {
    if (this.pending.has(key)) {
      return false;
    }
    this.pending.set(key, true);
    setTimeout(() => this.pending.delete(key), 10000);
    return true;
  }
};

// Skeleton screen loading (improve perceived performance)
function showSkeleton(container, count = 3) {
  container.innerHTML = Array(count).fill(`
    <div class="skeleton-item p-3 mb-2 rounded">
      <div class="skeleton-line skeleton-title mb-2"></div>
      <div class="skeleton-line skeleton-body"></div>
    </div>
  `).join('');
}

// Request timeout handling
function fetchWithTimeout(url, options = {}, timeout = 15000) {
  return Promise.race([
    fetch(url, options),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Request timed out, please try again later')), timeout)
    )
  ]);
}

// Scroll progress percentage
function updateScrollProgress() {
  const scrollTop = window.scrollY;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const progress = docHeight > 0 ? Math.min((scrollTop / docHeight) * 100, 100) : 0;
  const bar = document.querySelector('.scroll-progress-bar');
  if (bar) bar.style.width = progress + '%';
}

// Performance monitoring (Core Web Vitals)
function reportWebVitals({ name, delta, id }) {
  console.log(`[CWV] ${name}: ${delta.toFixed(2)}ms (id: ${id})`);
}

if (typeof window !== 'undefined' && 'PerformanceObserver' in window) {
  try {
    new PerformanceObserver(list => {
      list.getEntries().forEach(entry => {
        if (entry.entryType === 'navigation') {
          const ttfb = entry.responseStart - entry.requestStart;
          if (ttfb > 200) {
            console.warn(`[Performance] TTFB is slow: ${ttfb.toFixed(0)}ms`);
          }
        }
      });
    }).observe({ type: 'navigation', buffered: true });
  } catch (e) {
    // Performance monitoring not available, silently ignore
  }
}
