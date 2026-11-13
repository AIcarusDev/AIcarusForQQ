(() => {
  'use strict';
  const routes = new Set(['/', '/focus', '/chat', '/agent', '/tool-stats', '/token-stats', '/log', '/memory', '/memory/3d', '/stickers', '/settings', '/maintenance']);
  const cleanURL = input => {
    const url = new URL(input, location.href);
    url.searchParams.delete('_content');
    return url;
  };
  const isAppURL = url => url.origin === location.origin && routes.has(url.pathname);
  const isNormalClick = (event, link) => event.button === 0 && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey && !link.hasAttribute('download') && (!link.target || link.target === '_self');

  if (document.body.hasAttribute('data-content-page')) {
    // A directly opened content URL should still have normal application navigation.
    if (window.parent === window) {
      location.replace(cleanURL(location.href).href);
      return;
    }
    const send = message => parent.postMessage(message, location.origin);
    const publishHeading = () => send({
      type: 'app:heading',
      heading: document.querySelector('.topbar-heading')?.innerHTML,
      description: document.querySelector('.topbar-desc')?.textContent || '',
      title: document.title,
      page: document.body.dataset.page,
    });
    const heading = document.querySelector('.topbar-heading');
    if (heading) new MutationObserver(publishHeading).observe(heading, {childList:true, subtree:true, characterData:true});
    // Only the outer shell owns browser history; avoid duplicate iframe entries.
    const replaceContentState = history.replaceState.bind(history);
    for (const method of ['pushState', 'replaceState']) {
      history[method] = (...args) => {
        const result = replaceContentState(...args);
        send({type:'app:route', url:cleanURL(location.href).href, replace:method === 'replaceState'});
        return result;
      };
    }
    addEventListener('hashchange', () => send({type:'app:route', url:cleanURL(location.href).href}));
    addEventListener('storage', event => {
      if (event.key === 'afq_ui_theme' && event.newValue && typeof setTheme === 'function') setTheme(event.newValue);
    });
    document.addEventListener('click', event => {
      const link = event.target.closest('a[href]');
      if (!link || !isNormalClick(event, link)) return;
      const url = cleanURL(link.href);
      if (!isAppURL(url) || (url.pathname === location.pathname && url.search === cleanURL(location.href).search)) return;
      event.preventDefault();
      send({type:'app:navigate', url:url.href});
    });
    publishHeading();
    return;
  }

  const host = document.getElementById('shellContentHost');
  if (!host) return;
  let current = null;
  let pending = null;
  let generation = 0;
  let currentURL = null;
  let acceptedURL = location.href;
  const frames = new Set();
  const pageViews = new Map();
  const viewKey = url => url.pathname + url.search;
  function rememberView() {
    if (!current || !currentURL) return;
    const doc = current.contentDocument;
    const scroll = Array.from(doc.querySelectorAll('[data-scroll-key]'), el => ({
      key: el.dataset.scrollKey, top: el.scrollTop, left: el.scrollLeft,
    }));
    pageViews.set(viewKey(currentURL), { scroll, controls: structuredClone(current.contentWindow.capturePageView?.()) });
  }
  function restoreScroll(frame, url) {
    const saved = pageViews.get(viewKey(url));
    if (!saved) return;
    const containers = new Map(Array.from(frame.contentDocument.querySelectorAll('[data-scroll-key]'), el => [el.dataset.scrollKey, el]));
    for (const position of saved.scroll) {
      const el = containers.get(position.key);
      if (el) { el.scrollTop = position.top; el.scrollLeft = position.left; }
    }
  }
  const nextPaint = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

  function shareRuntime(frame) {
    try {
      frame.contentWindow.workbenchRuntime = {...window.workbenchRuntime};
      frame.contentWindow.dispatchEvent(new frame.contentWindow.Event('workbench-status'));
    } catch (_) { /* The frame may still be loading or redirecting. */ }
  }
  addEventListener('workbench-status', () => frames.forEach(shareRuntime));

  function applyHeading(frame) {
    const doc = frame.contentDocument;
    const heading = doc?.querySelector('.topbar-heading');
    if (!heading) return;
    document.querySelector('.topbar-heading').innerHTML = heading.innerHTML;
    const description = document.querySelector('.topbar-desc');
    description.textContent = doc.querySelector('.topbar-desc')?.textContent || '';
    description.hidden = doc.body.dataset.page === 'settings';
    document.title = doc.title;
    document.body.dataset.page = doc.body.dataset.page || '';
    document.querySelectorAll('.sidebar-nav a[href]').forEach(link => {
      const path = new URL(link.href).pathname;
      link.classList.toggle('active', path === currentURL.pathname || (path !== '/' && currentURL.pathname.startsWith(path + '/')));
    });
  }

  let leaveConfirmation = null;
  async function canLeave() {
    try {
      const dirty = current?.contentWindow.allDraftsDirty;
      if (typeof dirty !== 'function' || !dirty()) return true;
    } catch (_) { /* A server redirect may have replaced the content document. */ }
    if (leaveConfirmation) return leaveConfirmation;
    const previousFocus = document.activeElement;
    const overlay = document.createElement('div');
    overlay.className = 'webui-modal-overlay active';
    overlay.innerHTML = `
      <div class="webui-modal" role="dialog" aria-modal="true" aria-labelledby="leave-draft-title" aria-describedby="leave-draft-description">
        <div class="webui-modal-header"><h2 id="leave-draft-title">有未保存的修改</h2></div>
        <div class="webui-modal-body" id="leave-draft-description">离开后，当前页面尚未保存的修改将丢失。</div>
        <div class="webui-modal-footer">
          <button type="button" class="webui-modal-btn primary" data-leave="stay">继续编辑</button>
          <button type="button" class="webui-modal-btn danger" data-leave="discard">放弃修改并离开</button>
        </div>
      </div>`;
    leaveConfirmation = new Promise(resolve => {
      const finish = leave => {
        document.removeEventListener('keydown', onKeyDown, true);
        overlay.remove();
        leaveConfirmation = null;
        if (!leave && previousFocus?.isConnected) previousFocus.focus({ preventScroll: true });
        resolve(leave);
      };
      const onKeyDown = event => {
        if (event.key === 'Escape') {
          event.preventDefault(); event.stopImmediatePropagation(); finish(false);
        } else if (event.key === 'Tab') trapDialogTab(event, overlay);
      };
      overlay.querySelector('[data-leave="stay"]').onclick = () => finish(false);
      overlay.querySelector('[data-leave="discard"]').onclick = () => finish(true);
      document.body.appendChild(overlay);
      document.addEventListener('keydown', onKeyDown, true);
      overlay.querySelector('[data-leave="stay"]').focus();
    });
    return leaveConfirmation;
  }

  async function navigate(input, options = {}) {
    const url = cleanURL(input);
    if (!isAppURL(url)) { location.assign(url.href); return; }
    if (url.href === currentURL?.href && !pending) return;
    if (current && url.pathname === currentURL.pathname && url.search === currentURL.search && !pending) {
      const target = new URL(url.href);
      target.searchParams.set('_content', '1');
      current.contentWindow.history.replaceState(null, '', target.href);
      current.contentWindow.dispatchEvent(new current.contentWindow.PopStateEvent('popstate'));
      currentURL = url;
      if (options.history !== 'pop' && location.href !== url.href) history.pushState(null, '', url.href);
      acceptedURL = url.href;
      applyHeading(current);
      return;
    }
    if (!await canLeave()) {
      if (options.history === 'pop') history.pushState(null, '', acceptedURL);
      return;
    }
    const token = ++generation;
    rememberView();
    if (pending) { frames.delete(pending); pending.remove(); pending = null; }
    host.querySelector('.shell-content-error')?.remove();
    host.setAttribute('aria-busy', 'true');
    const frame = document.createElement('iframe');
    frame.className = 'shell-content-frame';
    frame.title = '页面内容';
    frame.setAttribute('aria-hidden', 'true');
    frame.inert = true;
    const target = new URL(url.href);
    target.searchParams.set('_content', '1');
    pending = frame;
    frames.add(frame);
    frame.addEventListener('load', async () => {
      try {
        if (frame.contentWindow.location.pathname === '/login' && (frame === current || frame === pending)) {
          location.assign(cleanURL(frame.contentWindow.location.href).href);
          return;
        }
        if (token !== generation || pending !== frame) return;
        if (!frame.contentDocument?.querySelector('.page-content')) throw new Error('页面加载失败');
        shareRuntime(frame);
        // Restore controls before the first request; keep the current page visible while loading.
        const savedView = pageViews.get(viewKey(url));
        const ready = typeof frame.contentWindow.initializePage === 'function'
          ? frame.contentWindow.initializePage(structuredClone(savedView?.controls))
          : frame.contentWindow.appPageReady;
        await Promise.race([
          Promise.resolve(ready),
          new Promise(resolve => setTimeout(resolve, 5000)),
        ]);
        await nextPaint();
        if (token !== generation || pending !== frame) return;
        restoreScroll(frame, url);
        const previous = current;
        current = frame;
        pending = null;
        currentURL = url;
        frame.classList.add('is-current');
        frame.removeAttribute('aria-hidden');
        frame.inert = false;
        frame.title = frame.contentDocument.title;
        frame.style.zIndex = '1';
        host.querySelector('.shell-content-loading')?.remove();
        if (options.history !== 'pop' && location.href !== url.href) history.pushState(null, '', url.href);
        acceptedURL = url.href;
        applyHeading(frame);
        host.setAttribute('aria-busy', 'false');
        if (previous) {
          previous.classList.remove('is-current');
          previous.inert = true;
          previous.setAttribute('aria-hidden', 'true');
          previous.style.pointerEvents = 'none';
          previous.style.zIndex = '2';
          if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
            const fade = previous.animate([{opacity:1},{opacity:0}], {duration:160, easing:'cubic-bezier(.2,0,.2,1)', fill:'forwards'});
            await fade.finished.catch(() => {});
          }
          frames.delete(previous);
          previous.remove();
        }
      } catch (error) {
        if (token !== generation) return;
        pending = null;
        frames.delete(frame);
        frame.remove();
        host.setAttribute('aria-busy', 'false');
        const note = document.createElement('p');
        note.className = 'shell-content-error';
        note.setAttribute('role', 'alert');
        note.textContent = '页面加载失败，请重新点击左侧入口。';
        host.appendChild(note);
      }
    });
    frame.src = target.href;
    host.appendChild(frame);
  }

  document.addEventListener('click', event => {
    const link = event.target.closest('a[href]');
    if (!link || !isNormalClick(event, link)) return;
    const url = cleanURL(link.href);
    if (!isAppURL(url)) return;
    event.preventDefault();
    const menu = document.getElementById('mobile-nav-toggle');
    if (menu?.getAttribute('aria-expanded') === 'true') menu.click();
    navigate(url.href);
  });
  addEventListener('popstate', () => navigate(location.href, {history:'pop'}));
  addEventListener('beforeunload', event => {
    try {
      if (typeof current?.contentWindow.allDraftsDirty === 'function' && current.contentWindow.allDraftsDirty()) {
        event.preventDefault();
        event.returnValue = '';
      }
    } catch (_) {}
  });
  addEventListener('message', event => {
    if (event.origin !== location.origin || !current || event.source !== current.contentWindow) return;
    const message = event.data;
    if (message?.type === 'app:navigate') {
      const url = cleanURL(message.url);
      if (isAppURL(url)) navigate(url.href);
    } else if (message?.type === 'app:route') {
      const url = cleanURL(message.url);
      if (url.pathname !== currentURL.pathname || !isAppURL(url)) return;
      currentURL = url;
      if (location.href !== url.href) history[message.replace ? 'replaceState' : 'pushState'](null, '', url.href);
      acceptedURL = url.href;
      applyHeading(current);
    } else if (message?.type === 'app:heading') {
      applyHeading(current);
    }
  });
  // Start the first content frame after the shell finishes loading. Embedded
  // browsers may install document observers while the outer page is parsing.
  const openInitialPage = () => navigate(location.href, {history:'pop'});
  if (document.readyState === 'complete') openInitialPage();
  else addEventListener('load', openInitialPage, {once:true});
})();
