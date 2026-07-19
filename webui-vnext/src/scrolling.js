const activeScrolls = new WeakMap();

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function reducedMotionRequested() {
  return globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;
}

export function cancelElementScroll(element) {
  activeScrolls.get(element)?.();
}

export function smoothScrollElement(
  element,
  targetTop,
  { minimumDuration = 280, maximumDuration = 680, onComplete } = {},
) {
  if (!element) return () => {};
  cancelElementScroll(element);

  const maximumTop = Math.max(0, element.scrollHeight - element.clientHeight);
  const destination = clamp(Number(targetTop) || 0, 0, maximumTop);
  const startTop = element.scrollTop;
  const distance = destination - startTop;

  if (Math.abs(distance) < 1 || reducedMotionRequested()) {
    element.scrollTop = destination;
    onComplete?.();
    return () => {};
  }

  const duration = clamp(Math.abs(distance) / 2.4, minimumDuration, maximumDuration);
  let animationFrame = 0;
  let startedAt = null;
  let cancelled = false;

  const cancel = () => {
    if (cancelled) return;
    cancelled = true;
    globalThis.cancelAnimationFrame(animationFrame);
    if (activeScrolls.get(element) === cancel) activeScrolls.delete(element);
  };

  const animate = (timestamp) => {
    if (cancelled) return;
    if (startedAt === null) startedAt = timestamp;
    const progress = Math.min(1, (timestamp - startedAt) / duration);
    const eased = 1 - ((1 - progress) ** 4);
    element.scrollTop = startTop + (distance * eased);
    if (progress < 1) {
      animationFrame = globalThis.requestAnimationFrame(animate);
      return;
    }
    element.scrollTop = destination;
    activeScrolls.delete(element);
    onComplete?.();
  };

  activeScrolls.set(element, cancel);
  animationFrame = globalThis.requestAnimationFrame(animate);
  return cancel;
}
