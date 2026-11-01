/* Shared settings workbench behavior. The existing settings payload remains authoritative. */
const workbenchSnapshots = new Map();
const workbenchFadingPanes = new Map();
function stackWorkbenchFades() {
  const panes = Array.from(workbenchFadingPanes.keys());
  panes.forEach((pane, index) => pane.style.zIndex = String(panes.length - index + 2));
}
function switchWorkbenchSection(sectionId, applySection) {
  if (sectionId === state.activeSection) return false;
  const incoming = document.getElementById('section-' + sectionId);
  // Returning to a fading pane makes it interactive immediately.
  workbenchFadingPanes.get(incoming)?.stop();
  const outgoing = document.getElementById('section-' + state.activeSection);
  const motion = state.raw && outgoing?.animate && !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!motion) {
    Array.from(workbenchFadingPanes.values()).forEach(entry => entry.stop());
    applySection();
    return true;
  }
  outgoing.classList.add('workbench-fading-pane');
  applySection();
  // Keep the incoming page opaque underneath; dissolve the old page in place.
  const animation = outgoing.animate([{opacity:1}, {opacity:0}], {
    duration:160,
    easing:'cubic-bezier(0.2, 0, 0.2, 1)',
    fill:'forwards',
  });
  const entry = {
    stop() {
      if (workbenchFadingPanes.get(outgoing) !== entry) return;
      workbenchFadingPanes.delete(outgoing);
      outgoing.classList.remove('workbench-fading-pane');
      outgoing.style.removeProperty('z-index');
      animation.cancel();
      stackWorkbenchFades();
    },
  };
  workbenchFadingPanes.set(outgoing, entry);
  stackWorkbenchFades();
  // Rapid navigation blends toward the last choice without queueing clicks.
  animation.finished.then(() => entry.stop(), () => entry.stop());
  return true;
}
function rememberWorkbenchControls(onlySection = null) {
  for (const [section] of SECTION_IDS) {
    if (onlySection && section !== onlySection) continue;
    const fields = Array.from(document.querySelectorAll(`#section-${section} input[id], #section-${section} select[id], #section-${section} textarea[id]`))
      .filter(field => section !== 'persona' || ['self_name', 'timezone'].includes(field.id));
    workbenchSnapshots.set(section, {
      fields: fields.map(field => ({id:field.id, value:field.value, checked:field.checked})),
      lists: section === 'qq_adapter' ? structuredClone(state.lists) : null,
    });
  }
}
function discardWorkbenchSection() {
  const section = state.activeSection;
  if (state.saving || !sectionHasChanges(section)) return;
  if (!confirm('撤销当前分区尚未保存的修改？其他分区的草稿会保留。')) return;
  if (section === 'providers') {
    state.providers = structuredClone(JSON.parse(state.lastSavedSignature).model_providers || {});
    clearProviderApiKeyInputs();
    renderProviderRows(state.raw?.api_keys || {});
    refreshProviderSelects();
  } else {
    const snapshot = workbenchSnapshots.get(section);
    for (const field of snapshot?.fields || []) {
      const target = document.getElementById(field.id);
      if (!target) continue;
      target.value = field.value;
      if (target.type === 'checkbox') target.checked = field.checked;
    }
    if (snapshot?.lists) {
      state.lists = structuredClone(snapshot.lists);
      renderList('private'); renderList('group');
    }
    document.querySelectorAll(`#section-${section} input[type="range"]`).forEach(input => input.dispatchEvent(new Event('input', {bubbles:true})));
    onVisionModeChange(); syncOptionalFields(); syncCognitionThinkingMode();
    updateTypingSpeedLabels(); onProviderChange();
  }
  document.querySelectorAll(`#section-${section} .settings-field-error`).forEach(el => el.remove());
  document.querySelectorAll(`#section-${section} [aria-invalid]`).forEach(el => el.removeAttribute('aria-invalid'));
  updateDirtyState();
}
function addWorkbenchModes(selectId, labels) {
  const select = document.getElementById(selectId);
  const card = select.closest('.model-card');
  const head = card.querySelector('.model-card-head');
  const group = document.createElement('div');
  group.className = 'segmented-modes';
  group.setAttribute('role', 'group');
  group.setAttribute('aria-label', selectId === 'vision_mode' ? '图片理解模式' : '视频理解配置方式');
  for (const [value, title] of labels) {
    const button = document.createElement('button');
    button.type = 'button'; button.textContent = title;
    button.dataset.mode = value; button.dataset.select = selectId;
    button.addEventListener('click', () => {
      select.value = value;
      select.dispatchEvent(new Event('change', {bubbles:true}));
      if (selectId === 'vision_mode') onVisionModeChange();
      else syncOptionalFields();
      updateDirtyState();
    });
    group.appendChild(button);
  }
  const originalField = select.closest('.field');
  select.hidden = true;
  head.append(select, group);
  originalField.remove();
}
function wrapWorkbenchAdvanced(container, keepCount) {
  const children = Array.from(container.children);
  const extra = children.slice(keepCount);
  if (!extra.length) return;
  const details = document.createElement('details');
  details.className = 'model-advanced';
  const summary = document.createElement('summary');
  summary.textContent = '高级参数 · 提示词、生成与缓存设置';
  const content = document.createElement('div');
  content.append(...extra); details.append(summary, content); container.appendChild(details);
}
function syncWorkbenchModes() {
  if (!document.querySelector('.segmented-modes')) return;
  document.querySelectorAll('[data-select][data-mode]').forEach(button => {
    button.setAttribute('aria-pressed', String(document.getElementById(button.dataset.select).value === button.dataset.mode));
  });
  document.getElementById('card-vision-bridge-body').hidden = value('vision_mode') !== 'bridge';
  document.getElementById('card-video-understanding-body').hidden = value('video_understanding_connection_mode') === 'unconfigured';
}
function toggleProxyClear(id) {
  if (state.saving || !state.raw) return;
  setValue(id + '_clear', value(id + '_clear') === '1' ? '' : '1');
  updateDirtyState();
}
function syncProxyClearButtons() {
  for (const id of ['proxy_openai', 'proxy_tavily']) {
    const pending = value(id + '_clear') === '1';
    const input = document.getElementById(id);
    const button = document.getElementById(id + '_clear_button');
    button.textContent = pending ? '撤销清除' : '清除代理';
    button.disabled = !state.raw || state.saving || (!pending && !input.value.trim() && !input.placeholder.startsWith('已设置'));
    input.readOnly = pending;
    input.setAttribute('aria-describedby', id + '_clear_status');
    document.getElementById(id + '_clear_status').textContent = pending ? '保存本分区后清除' : '';
  }
}
function workbenchTextDrafts() {
  if (!state.raw) return [];
  const drafts = [];
  if (guardianHasUnsavedChanges()) drafts.push({section:'persona', label:'监护人介绍', field:'guardian_info'});
  if (document.getElementById('persona-edit').style.display === 'block' && rawValue('persona-edit') !== personaOriginal)
    drafts.push({section:'persona', label:'角色文本', field:'persona-edit'});
  if (document.getElementById('qq-social-style-edit').style.display === 'block' && rawValue('qq-social-style-edit') !== qqSocialStyleOriginal)
    drafts.push({section:'persona', label:'QQ 社交风格', field:'qq-social-style-edit'});
  if (agentPromptHasUnsavedChanges()) drafts.push({section:'agent_prompt', label:'认知指令', field:'agent-prompt-instruction'});
  return drafts;
}
function jumpToWorkbenchDraft(draft) {
  showSection(draft.section);
  const field = document.getElementById(draft.field);
  field.scrollIntoView({block:'center',behavior:'auto'});
  field.focus({preventScroll:true});
}
function workbenchIcon(name, filled = false) {
  const icon=document.createElement('span');
  icon.className='workbench-icon';
  icon.setAttribute('aria-hidden','true');
  icon.style.setProperty('--workbench-icon', `url('/static/vendor/tabler/${filled ? 'filled/' : ''}${name}.svg')`);
  return icon;
}
function refreshWorkbench() {
  syncProxyClearButtons();
  const ttsFields=document.getElementById('workbenchTtsFields');
  if(ttsFields) ttsFields.hidden=!checked('tts_enabled');
  const title = SECTION_IDS.find(([id]) => id === state.activeSection)?.[1] || '设置';
  document.getElementById('workbenchSectionTitle').textContent = title;
  document.querySelector('.settings-shell').dataset.section = state.activeSection;
  const textDrafts = workbenchTextDrafts();
  const dirtySections = state.lastSavedSignature ? SECTION_IDS.filter(([id]) => sectionHasChanges(id) || textDrafts.some(draft => draft.section === id)) : [];
  document.querySelectorAll('[data-section-button]').forEach(button => {
    button.querySelector('.draft-marker').hidden = !dirtySections.some(([id]) => id === button.dataset.sectionButton);
  });
  const others = dirtySections.filter(([id]) => id !== state.activeSection);
  const draftLinks = document.getElementById('workbenchOtherDrafts');
  draftLinks.replaceChildren();
  for (const [section, name] of others) {
    if (textDrafts.some(draft => draft.section === section) && !sectionHasChanges(section)) continue;
    const button = document.createElement('button');
    button.type='button'; button.className='workbench-draft-link'; button.textContent=`${name}有草稿`;
    button.onclick=()=>showSection(section); draftLinks.appendChild(button);
  }
  for (const draft of textDrafts) {
    const button = document.createElement('button');
    button.type='button'; button.className='workbench-draft-link'; button.textContent=`${draft.label}未保存`;
    button.onclick=()=>jumpToWorkbenchDraft(draft); draftLinks.appendChild(button);
  }
  const localDrafts = textDrafts.filter(draft => draft.section === state.activeSection);
  if (state.raw && !state.saving) {
    if (localDrafts.length) setStatus(localDrafts.map(draft => draft.label).join('、') + '未保存');
    else if (!SECTION_PATHS[state.activeSection]) {
      const status = {
        security:'访问保护：' + (document.getElementById('webuiAuthStatus')?.textContent || '读取中'),
        self_image:'图片上传后自动保存', cache:'清理操作通过页内按钮执行',
        agent_prompt:'认知指令使用页内按钮保存', workspace:'电脑设置使用页内按钮保存',
      };
      setStatus(status[state.activeSection] || '使用页内操作');
    } else if (!hasUnsavedChanges() && textDrafts.length) setStatus('其他分区有文本草稿');
  }
  const discard = document.getElementById('workbenchDiscard');
  discard.hidden = !SECTION_PATHS[state.activeSection];
  discard.disabled = state.saving || !sectionHasChanges(state.activeSection);
  const runtime = window.workbenchRuntime || {};
  for (const [id, flag, positive, negative] of [
    ['workbenchWebState',runtime.webConnected,'网页已连接','网页连接中断'],
    ['workbenchCoreState',runtime.coreRunning,'核心运行中','核心未运行'],
    ['workbenchQQState',runtime.qqConnected,'QQ 已连接','QQ 未连接'],
  ]) {
    const el = document.getElementById(id);
    if (!el) continue;
    const label=flag == null ? positive.split('已')[0].replace('运行中','') + '状态读取中' : flag ? positive : negative;
    const iconName={workbenchWebState:'world',workbenchCoreState:'settings',workbenchQQState:'bell'}[id];
    el.replaceChildren(workbenchIcon(iconName),document.createTextNode(label));
    el.dataset.tone = flag == null ? '' : flag ? 'ok' : 'warn';
  }
  const context = document.querySelector('.workbench-context');
  const contextual = ['model','providers','submodels','tts'].includes(state.activeSection);
  context.hidden = !contextual;
  const summary = document.getElementById('workbenchSavedSummary');
  const recovery = document.getElementById('workbenchRecovery');
  const note = document.getElementById('workbenchContextNote');
  summary.replaceChildren(); recovery.replaceChildren();
  note.textContent = '';
  const intro = document.querySelector('.workbench-context-content>.hint');
  intro.textContent = '配置完整不代表连接已通过检查。';
  if (contextual && state.lastSavedSignature) {
    const saved = JSON.parse(state.lastSavedSignature);
    const configured = cfg => !!(cfg?.provider && cfg?.model && saved.model_providers?.[cfg.provider]);
    const video = saved.video_understanding;
    let rows;
    if (state.activeSection === 'submodels') rows = [
      ['图片理解', saved.vision_bridge.enabled ? (configured(saved.vision_bridge) ? '独立视觉桥' : '配置待补全') : saved.vision ? '主模型处理' : '已关闭'],
      ['视频理解', video.connection_mode === 'unconfigured' ? '未配置' : video.model && (video.base_url || saved.model_providers?.[video.provider]) ? '已配置' : '配置待补全'],
      ['工具守门', !saved.tool_execution_guard.enabled ? '已关闭' : configured(saved.tool_execution_guard) ? '已配置' : '配置待补全'],
      ['上下文压缩', configured(saved.cognition_compression) ? '已配置' : '配置待补全'],
    ];
    else if (state.activeSection === 'providers') rows = [['供应商数量',String(Object.keys(saved.model_providers || {}).length)]];
    else if (state.activeSection === 'model') rows = [['主模型',configured(saved) ? '已配置' : '配置待补全']];
    else rows = [['语音服务',saved.tts.enabled ? '已启用配置' : '已关闭'],['监听端口',String(saved.tts.port)]];
    for (const [label, text] of rows) {
      const row=document.createElement('div'), dt=document.createElement('dt'), dd=document.createElement('dd');
      const iconName={'图片理解':'photo','视频理解':'video','工具守门':'shield-check','上下文压缩':'database','供应商数量':'database','主模型':'box','语音服务':'volume','监听端口':'plug'}[label];
      if(iconName) dt.appendChild(workbenchIcon(iconName));
      dt.appendChild(document.createTextNode(label)); dd.textContent=text;
      dd.dataset.tone=text==='配置待补全'?'warn':text==='已配置'?'ok':'';
      row.append(dt,dd); summary.appendChild(row);
    }
    if (state.activeSection === 'tts') {
      note.textContent='外部语音程序需使用相同的连接地址、端口和口令。';
    } else {
      const missingProvider=!saved.provider || !saved.model_providers?.[saved.provider];
      if (missingProvider) {
        const heading=document.createElement('strong'); heading.append(workbenchIcon('alert-circle'),document.createTextNode('主模型连接待完善'));
        const description=document.createElement('p'); description.textContent='先保存供应商，再到主模型中选择。';
        const button=document.createElement('button'); button.type='button'; button.className='btn-secondary';
        const target=state.activeSection==='providers'?'model':'providers';
        button.textContent=target==='model'?'前往主模型':'前往模型供应商'; button.appendChild(workbenchIcon('chevron-right')); button.onclick=()=>showSection(target);
        recovery.append(heading,description,button);
      }
      note.textContent=sectionHasChanges(state.activeSection)?'右侧为已保存配置，当前草稿尚未生效。':'';
    }
    if (runtime.coreRunning === false) note.textContent += (note.textContent ? ' ' : '') + '核心未运行，配置保存后启动生效。';
  }
  recovery.hidden=!recovery.childElementCount;
  note.hidden=!note.textContent;
  syncWorkbenchModes();
}
function initWorkbench() {
  for (const card of document.querySelectorAll('#section-submodels .model-card')) {
    const head = card.querySelector('.model-card-head');
    head.removeAttribute('onclick');
  }
  const visionTitle = document.querySelector('#card-vision-bridge h3');
  visionTitle.textContent = '图片理解';
  document.querySelector('#card-vision-bridge .card-desc').textContent = '选择关闭、由主模型处理，或使用独立视觉模型。';
  document.querySelector('#card-video-understanding .card-desc').textContent = '不使用视频理解时，无需配置专用模型。';
  document.querySelector('#card-tool-guard .card-desc').textContent = '执行工具前，重新判断外界变化是否影响原计划。';
  document.querySelector('#card-compression .card-desc').textContent = '压缩较长的上下文，减少后续输入占用。';
  document.querySelector('#card-compression .card-badge').textContent='必需';
  document.querySelector('#section-submodels .section-head p').textContent='专用任务独立配置，不使用的能力可以关闭。';
  addWorkbenchModes('vision_mode', [['off','关闭'],['native','主模型处理'],['bridge','独立视觉桥']]);
  addWorkbenchModes('video_understanding_connection_mode', [['unconfigured','暂不配置'],['provider','使用供应商'],['explicit','专用连接']]);
  const guardBody = document.getElementById('card-tool-guard-body');
  guardBody.dataset.workbench='true';
  document.querySelector('#card-tool-guard .model-card-head').appendChild(guardBody.firstElementChild);
  for (const [id, count] of [['visionBridgeFields',1],['card-tool-guard-body',1],['card-compression-body',1]])
    wrapWorkbenchAdvanced(document.getElementById(id),count);
  document.querySelector('#card-tool-guard .model-advanced summary').textContent='高级参数 · 图片输入、温度与输出长度';
  document.querySelector('#card-compression .model-advanced summary').textContent='高级参数 · 温度、输出长度与思维链';
  document.querySelectorAll('#section-submodels .field').forEach(field => {
    const label = field.querySelector('label');
    if (!label) return;
    if (label.textContent.trim() === 'Temperature') label.textContent='温度';
    if (label.textContent.trim() === 'max_output_tokens') label.textContent='最大输出 Token';
    const control=field.querySelector('input[id],select[id],textarea[id]');
    if(control) label.htmlFor=control.id;
  });
  document.getElementById('guard_enabled').setAttribute('aria-label','启用工具执行前守门');
  const ttsGroup=document.querySelector('#section-tts .group');
  const ttsFields=document.createElement('div'); ttsFields.id='workbenchTtsFields';
  while (ttsGroup.children.length > 1) ttsFields.appendChild(ttsGroup.children[1]);
  ttsGroup.appendChild(ttsFields);
  document.getElementById('tts_enabled').addEventListener('change',()=>{
    ttsFields.hidden=!checked('tts_enabled');
  });
  const identityRow=document.getElementById('self_name').closest('.row');
  const guardianGroup=identityRow.closest('.group');
  const identity=document.createElement('div'); identity.className='group';
  const identityLabel=document.createElement('div'); identityLabel.className='identity-save-scope';
  identityLabel.textContent='名称与时区 · 底部保存';
  identity.append(identityLabel,identityRow); guardianGroup.before(identity);
  for (const selector of ['#guardian_info','#persona-edit','#qq-social-style-edit']) {
    const field=document.querySelector(selector);
    const group=field.closest('.group');
    const title=group.querySelector('.group-title');
    const badge=title?.querySelector('.subtle') || document.createElement('span');
    badge.className='save-scope-badge'; badge.textContent='单独保存';
    (title || field.closest('.field').querySelector('label')).appendChild(badge);
  }
  const pageIcons = {
    providers:'database', model:'category', submodels:'stack', memory:'database',
    persona:'id', agent_prompt:'settings', self_image:'photo', cache:'database',
    workspace:'settings', qq_adapter:'bell', tts:'microphone',
    'qq-supervisor':'settings', auxiliary:'settings', advanced:'settings', security:'shield',
  };
  document.querySelectorAll('.settings-section').forEach(section => {
    const title=section.querySelector('.section-head h2');
    if (title) {
      const icon=workbenchIcon(pageIcons[section.id.replace('section-','')] || 'settings',true);
      icon.classList.add('page-title-icon'); title.prepend(icon);
    }
  });
  const compressionTitle=document.querySelector('#card-compression h3');
  compressionTitle.appendChild(document.querySelector('#card-compression .card-badge'));
  document.querySelector('.workbench-nav-title').prepend(workbenchIcon('settings',true));
  document.querySelector('.workbench-context-content h2').prepend(workbenchIcon('database',true));
  document.querySelector('#workbenchContextPanel>summary').prepend(workbenchIcon('settings',true));
  const functionalIcon = title => {
    if (/图片|形象/.test(title)) return 'photo';
    if (/视频/.test(title)) return 'video';
    if (/模型选择|供应商/.test(title)) return 'box';
    if (/压缩|记忆/.test(title)) return 'database';
    if (/守门|保护|安全|密码/.test(title)) return 'shield-check';
    if (/角色|Persona|监护/.test(title)) return 'user-circle';
    if (/社交/.test(title)) return 'message-circle';
    if (/搜索|联网/.test(title)) return 'world';
    if (/电脑|安装/.test(title)) return 'device-desktop';
    if (/语音/.test(title)) return 'volume';
    return 'adjustments';
  };
  document.querySelectorAll('.settings-section .group-title h3, .settings-section .model-card-title h3').forEach(title => {
    title.prepend(workbenchIcon(functionalIcon(title.textContent)));
  });
  document.querySelectorAll('.settings-section .group, .settings-section .model-card').forEach(panel => {
    if (panel.parentElement.closest('.group, .model-card')) return;
    const title=(panel.querySelector('.group-title h3, .model-card-title h3, .identity-save-scope, summary, .toggle-copy strong')?.textContent || '').trim();
    const section=panel.closest('.settings-section').id;
    let tone='generation';
    if (/模型选择|供应商|图片理解|视频理解|连接|代理|搜索|天气|浏览器/.test(title)) tone='connection';
    else if (/保护|守门|密码|安全/.test(title)) tone='protection';
    else if (/记忆|事件提取|压缩|故事线/.test(title)) tone='memory';
    else if (/角色|名称|身份|监护|社交/.test(title)) tone='identity';
    else if (section==='section-persona' || section==='section-agent_prompt') tone='identity';
    else if (section==='section-security') tone='protection';
    else if (section==='section-providers' || section==='section-tts' || section==='section-auxiliary') tone='connection';
    else if (section==='section-memory') tone='memory';
    panel.dataset.panelTone=tone;
  });
  const narrow=window.matchMedia('(max-width:1096px)');
  const updateContext=()=>{document.getElementById('workbenchContextPanel').open=!narrow.matches;};
  updateContext(); narrow.addEventListener('change',updateContext);
  window.addEventListener('workbench-status',refreshWorkbench);
  window.addEventListener('popstate',()=>{
    const section=location.hash.slice(1);
    showSection(SECTION_IDS.some(([id])=>id===section)?section:'model',false);
  });
  document.addEventListener('click',event=>{
    document.querySelectorAll('.rail-menu[open]').forEach(menu=>{if(!menu.contains(event.target))menu.open=false;});
  });
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape')document.querySelectorAll('.rail-menu[open]').forEach(menu=>menu.open=false);
  });
}
