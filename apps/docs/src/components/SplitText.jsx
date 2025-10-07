import React from 'react';
import { motion } from 'framer-motion';

const SplitText = ({ 
  text, 
  delay = 0, 
  duration = 0.05,
  animationFrom = { opacity: 0, y: 20 },
  animationTo = { opacity: 1, y: 0 },
  className = '',
  easing = [0.42, 0, 0.58, 1] 
}) => {
  const letters = text.split('');

  return (
    <span className={className} style={{ display: 'inline-block' }}>
      {letters.map((letter, index) => (
        <motion.span
          key={index}
          initial={animationFrom}
          animate={animationTo}
          transition={{
            duration: duration,
            delay: delay + index * 0.03,
            ease: easing,
          }}
          style={{ display: 'inline-block', whiteSpace: 'pre' }}
        >
          {letter}
        </motion.span>
      ))}
    </span>
  );
};

export default SplitText;
