/** Shared native-details animation, including the dynamic pipeline controls. */
export function setupDisclosures(root = document) {
    const animations = new WeakMap();
    const prepare = () => root.querySelectorAll('details').forEach(details => {
        if (!animations.has(details)) details.querySelector('summary')?.setAttribute('aria-expanded', String(details.open));
        if (details.querySelector(':scope > .disclosure-content')) return;
        const group = details.getAttribute('name');
        if (group) { details.dataset.group = group; details.removeAttribute('name'); }
        const content = document.createElement('div');
        content.className = 'disclosure-content';
        [...details.childNodes].filter(node => node !== details.querySelector('summary')).forEach(node => content.append(node));
        details.append(content);
    });
    const toggle = (details, open) => {
        const summary = details.querySelector(':scope > summary');
        const from = details.getBoundingClientRect().height;
        animations.get(details)?.cancel();
        animations.delete(details);
        details.style.overflow = '';
        details.dataset.expanded = String(open);
        summary.setAttribute('aria-expanded', String(open));
        if (!open && details.contains(document.activeElement) && document.activeElement !== summary) summary.focus();
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { details.open = open; return; }
        details.open = true;
        const to = open ? summary.getBoundingClientRect().height + details.querySelector(':scope > .disclosure-content').getBoundingClientRect().height
            : summary.getBoundingClientRect().height;
        details.style.overflow = 'hidden';
        const animation = details.animate([{height: `${from}px`}, {height: `${to}px`}],
            {duration: 240, easing: 'cubic-bezier(.2,.7,.3,1)'});
        animations.set(details, animation);
        animation.onfinish = () => { details.open = open; details.style.overflow = ''; animations.delete(details); };
    };
    prepare();
    const observer = new MutationObserver(prepare);
    observer.observe(root.body || root, {childList: true, subtree: true, attributes: true, attributeFilter: ['open']});
    root.addEventListener('click', event => {
        const summary = event.target.closest('summary');
        if (!summary || summary.parentElement.tagName !== 'DETAILS') return;
        event.preventDefault();
        const details = summary.parentElement;
        const open = animations.has(details) ? details.dataset.expanded !== 'true' : !details.open;
        if (open && details.dataset.group) {
            root.querySelectorAll('details[open]').forEach(other => {
                if (other !== details && other.dataset.group === details.dataset.group) toggle(other, false);
            });
        }
        toggle(details, open);
    });
    return observer;
}
