import { useState, useEffect } from 'react';
import './styles/TextAnimation.css';

interface TextAnimationProps {
  words: string[];
  delayTime?: number;
  animationTime?: number;
  height?: number;
  fontSize?: number;
}

export default function TextAnimation({
  words,
  delayTime = 2500,
  animationTime = 500,
  height = 40,
  fontSize = 16
}: TextAnimationProps) {
  const [wordIndex, setWordIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setIsAnimating(true);

      // after animation completes, update the word index
      setTimeout(() => {
        setWordIndex((prevIndex) => (prevIndex + 1) % words.length);
        setIsAnimating(false);
      }, animationTime);
    }, delayTime);

    return () => clearInterval(interval);
  }, [words, delayTime, animationTime]);

  return (
    <span
      className="text-animation"
      style={{
        '--animation-duration': `${animationTime}ms`,
        '--word-height': `${height}px`,
        '--container-height': `${height * 2}px`,
        '--font-size': `${fontSize}px`
      } as React.CSSProperties}
    >
        <div className={`animation-container ${isAnimating ? 'animate' : ''}`}>
            <div className="current-word">
                {words[wordIndex]}
            </div>
            <div className="next-word">
                {words[(wordIndex + 1) % words.length]}
            </div>
        </div>
    </span>
  );
}
