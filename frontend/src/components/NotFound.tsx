import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router";

const REDIRECT_SECONDS = 5;

export default function NotFound() {
  const navigate = useNavigate();
  const [secondsLeft, setSecondsLeft] = useState(REDIRECT_SECONDS);

  useEffect(() => {
    const countdown = setInterval(() => {
      setSecondsLeft((prev) => Math.max(prev - 1, 0));
    }, 1000);

    const redirect = setTimeout(() => {
      navigate("/", { replace: true });
    }, REDIRECT_SECONDS * 1000);

    return () => {
      clearInterval(countdown);
      clearTimeout(redirect);
    };
  }, [navigate]);

  return (
    <div className="App" style={{ padding: "40px", textAlign: "center" }}>
      <h1>Page not found</h1>
      <p>We couldn&apos;t find that page. Redirecting to Live Location in {secondsLeft} seconds.</p>
      <div style={{ marginTop: "20px" }}>
        <Link to="/">Go to Live Location now</Link>
      </div>
    </div>
  );
}
