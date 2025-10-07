import React from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Clock, Brain, Zap } from 'lucide-react';

const stats = [
  { number: '4', label: 'Supported Subjects', icon: BookOpen },
  { number: '60-90s', label: 'Video Length', icon: Clock },
  { number: '9', label: 'AI Models', icon: Brain },
  { number: '2-4min', label: 'Generation Time', icon: Zap },
];

export default function Stats() {
  return (
    <section className="stats-section">
      <div className="container">
        <motion.h2
          className="stats-title"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          By the Numbers
        </motion.h2>
        <div className="stats-grid">
          {stats.map((stat, index) => (
            <motion.div
              key={index}
              className="stat-card"
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
            >
              <stat.icon size={48} strokeWidth={1.5} style={{ marginBottom: '0.5rem', opacity: 0.8 }} />
              <div className="stat-number">{stat.number}</div>
              <div className="stat-label">{stat.label}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
