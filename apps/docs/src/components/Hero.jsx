import React from 'react';
import { motion } from 'framer-motion';
import SplitText from './SplitText';
import Prism from './Prism';
import { Calculator, FlaskConical, Zap, Code } from 'lucide-react';

export default function Hero() {
  return (
    <section className="hero-section">
      {/* Prism Background */}
      <div className="hero-prism-container">
        <Prism />
      </div>

      {/* Hero Content */}
      <div className="hero-content">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          {/* Title with SplitText Animation */}
          <h1 className="hero-title">
            <SplitText text="alfa" delay={0.2} />
          </h1>

          {/* Subtitle */}
          <p className="hero-subtitle">
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8, duration: 0.6 }}
            >
              from problem to ai generated explainer animations
            </motion.span>
          </p>

          {/* Subject Badges */}
          <motion.div
            className="subject-badges"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2, duration: 0.6 }}
          >
            <span className="subject-badge">
              <Calculator size={16} /> Math
            </span>
            <span className="subject-badge">
              <FlaskConical size={16} /> Chemistry
            </span>
            <span className="subject-badge">
              <Zap size={16} /> Physics
            </span>
            <span className="subject-badge">
              <Code size={16} /> CS
            </span>
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            className="cta-buttons"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.4, duration: 0.6 }}
          >
            <a href="/alfa/docs/intro" className="cta-button cta-primary">
              Get Started
            </a>
            <a href="https://github.com/OZIOisgood/alfa" className="cta-button cta-secondary" target="_blank" rel="noopener noreferrer">
              GitHub
            </a>
          </motion.div>

          {/* Promo Video */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 1.6, duration: 0.8 }}
            className="video-container"
          >
            <video 
              controls 
              autoPlay 
              muted 
              loop
            >
              <source src="/alfa/assets/alfa.promo.v1.mp4" type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}
