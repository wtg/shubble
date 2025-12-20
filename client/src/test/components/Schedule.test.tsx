import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Schedule from '../../components/Schedule';

describe('Schedule Component', () => {
  it('renders without crashing', () => {
    const mockSetSelectedRoute = () => {};
    render(<Schedule selectedRoute={null} setSelectedRoute={mockSetSelectedRoute} />);
    expect(document.body).toBeTruthy();
  });

  it('accepts route selection props', () => {
    const mockSetSelectedRoute = () => {};
    const { rerender } = render(
      <Schedule selectedRoute="NORTH" setSelectedRoute={mockSetSelectedRoute} />
    );

    // Rerender with different route
    rerender(<Schedule selectedRoute="WEST" setSelectedRoute={mockSetSelectedRoute} />);
    expect(document.body).toBeTruthy();
  });
});
