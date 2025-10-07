import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Mic, Sparkles, Zap, ImageIcon, Brain } from 'lucide-react';

const FeatureList = [
  {
    title: 'Multi-Subject Support',
    icon: BookOpen,
    description: (
      <>
        Generate educational videos for <strong>Mathematics, Chemistry, Physics, and Computer Science</strong>. 
        Each subject has specialized prompts and visualizations.
      </>
    ),
  },
  {
    title: 'AI Voice Narration',
    icon: Mic,
    description: (
      <>
        Automatic voice-over generation using <strong>Google Gemini TTS</strong>. 
        Natural-sounding narration synchronized perfectly with animations.
      </>
    ),
  },
  {
    title: 'Professional Animations',
    icon: Sparkles,
    description: (
      <>
        Powered by <strong>Manim</strong> – the same animation engine used by 3Blue1Brown. 
        Beautiful, smooth, and mathematically accurate visualizations.
      </>
    ),
  },
  {
    title: 'Lightning Fast',
    icon: Zap,
    description: (
      <>
        Generate a complete 60-90 second educational video in just <strong>2-4 minutes</strong>. 
        Perfect for quick explanations and study materials.
      </>
    ),
  },
  {
    title: 'Quality Options',
    icon: ImageIcon,
    description: (
      <>
        Choose from <strong>480p, 720p, 1080p, or 4K</strong> rendering. 
        Balance between speed and quality based on your needs.
      </>
    ),
  },
  {
    title: '9 LLM Models',
    icon: Brain,
    description: (
      <>
        Support for <strong>Gemini, GPT-4, Claude</strong> and more. 
        Choose the best model for your content generation needs.
      </>
    ),
  },
];

function Feature({ icon: Icon, title, description, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ delay: index * 0.1, duration: 0.6 }}
    >
      <div className="feature-card">
        <div className="feature-icon">
          <Icon size={40} strokeWidth={1.5} />
        </div>
        <h3 className="feature-title">{title}</h3>
        <p className="feature-description">{description}</p>
      </div>
    </motion.div>
  );
}

export default function Features() {
  return (
    <section className="features-section">
      <div className="container">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="features-title">Why Choose ALFA?</h2>
        </motion.div>

        <div className="features-grid">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} index={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
