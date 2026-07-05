// Identical header on every view — styled after itamarweiss.com's site header
// (Doto dot-matrix title, hairline bottom border, plain text nav).
export function Header() {
  return (
    <header className="site-header">
      <div className="wrapper">
        <a className="site-title" href="#/">
          FPV Dataset
        </a>
        <nav className="site-nav">
          <a href="#/">Videos</a>
          <a href="https://itamarweiss.com" rel="noreferrer">
            itamarweiss.com
          </a>
        </nav>
      </div>
    </header>
  );
}
