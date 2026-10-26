'use strict';
(() => {
  const content = window.GEOBIASET_CONTENT;
  const scenes = window.GEOBIASET_EXAMPLES;
  const groups = ['Black', 'East Asian', 'Indian', 'Latino', 'Middle Eastern', 'Southeast Asian', 'White'];
  const groupColors = { Black:'#e34f51', 'East Asian':'#eea524', Indian:'#d8bc31', Latino:'#31a87b', 'Middle Eastern':'#20a79b', 'Southeast Asian':'#438fc6', White:'#9062c7' };
  const state = { scene: 0, gender: 'Male', attire: 'Casual', group: 'Black' };
  const $ = (id) => document.getElementById(id);
  $('abstract-text').textContent = content.abstract;
  $('bibtex-code').textContent = content.citation;

  const slider = $('comparison-slider');
  slider.addEventListener('input', () => {
    slider.parentElement.style.setProperty('--split', `${slider.value}%`);
    slider.setAttribute('aria-valuetext', `${Number(slider.value).toFixed(1)} percent original scene visible`);
  });

  const coordinates = (scene) => `${Math.abs(scene.lat).toFixed(1)}° ${scene.lat >= 0 ? 'N' : 'S'} · ${Math.abs(scene.lon).toFixed(1)}° ${scene.lon >= 0 ? 'E' : 'W'}`;
  const dialog = $('image-dialog');
  let previouslyFocused = null;
  function openImage(image, caption, alt) {
    previouslyFocused = document.activeElement;
    $('dialog-image').src = image;
    $('dialog-image').alt = alt;
    $('dialog-caption').textContent = caption;
    $('open-image').href = image;
    dialog.showModal();
    document.body.classList.add('dialog-open');
    $('close-dialog').focus();
  }
  $('close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('dialog-open');
    previouslyFocused?.focus();
  });

  // A figure-like opening comparison: real source images, with no inferred locations.
  [scenes[0], scenes[3]].forEach((scene) => {
    const items = [{ group:'Original', image:scene.original }, ...groups.map(group => ({ group, image:scene.variants.find(v => v.group === group && v.gender === 'Male' && v.attire === 'Casual').image }))];
    items.forEach((item) => {
      const figure = document.createElement('figure');
      figure.className = 'teaser-card';
      if (groupColors[item.group]) figure.style.setProperty('--group-color', groupColors[item.group]);
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-label', `Inspect ${item.group} example from ${scene.title}`);
      const image = document.createElement('img');
      image.src = item.image;
      image.width = 1000;
      image.height = 667;
      image.alt = item.group === 'Original' ? `Original background in ${scene.title}.` : `${scene.title}: male subject annotated ${item.group}, with unconditioned attire. The original GPS label is retained.`;
      image.decoding = 'async';
      const caption = `${scene.title} · ${item.group} · ${coordinates(scene)}${item.group === 'Original' ? '' : ' · Male · Unconditioned attire'}`;
      button.addEventListener('click', () => openImage(item.image, caption, image.alt));
      button.append(image);
      const label = document.createElement('figcaption');
      label.textContent = item.group;
      figure.append(button, label);
      $('teaser-grid').append(figure);
    });
    const rowCaption = document.createElement('div');
    rowCaption.className = 'teaser-row-caption';
    const location = document.createElement('strong');
    location.textContent = `${scene.title} · ${coordinates(scene)}`;
    const annotation = document.createElement('span');
    annotation.textContent = 'Male · Unconditioned attire · Fixed GPS label';
    rowCaption.append(location, annotation);
    $('teaser-grid').append(rowCaption);
  });

  function renderComparison() {
    const scene = scenes[state.scene];
    const variant = scene.variants.find(v => v.group === state.group && v.gender === state.gender && v.attire === state.attire);
    const attireName = state.attire === 'Casual' ? 'Unconditioned' : 'Race-associated';
    $('comparison-original-image').src = scene.original;
    $('comparison-original-image').alt = `Original background in ${scene.title}.`;
    $('comparison-edited-image').src = variant.image;
    $('comparison-edited-image').alt = `${scene.title}: ${state.gender.toLowerCase()} subject annotated ${state.group}, ${attireName.toLowerCase()} attire. Original GPS label retained.`;
    $('comparison-edited-label').textContent = `${state.group} · ${state.gender} · ${attireName}`;
  }
  $('comparison-group').addEventListener('change', () => {
    state.group = $('comparison-group').value;
    renderComparison();
  });

  scenes.forEach((scene, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'scene-button';
    button.setAttribute('aria-pressed', String(index === state.scene));
    button.setAttribute('aria-label', `Show dataset examples from ${scene.title}`);
    const image = document.createElement('img');
    image.src = scene.original;
    image.alt = '';
    image.loading = 'lazy';
    image.width = 500;
    image.height = 333;
    const label = document.createElement('span');
    label.append(document.createTextNode(scene.title));
    const check = document.createElement('i');
    check.textContent = '✓';
    check.setAttribute('aria-hidden', 'true');
    label.append(check);
    button.append(image, label);
    button.addEventListener('click', () => { state.scene = index; renderGallery(); });
    $('scene-picker').append(button);
  });

  function renderGallery() {
    const scene = scenes[state.scene];
    $('scene-title').textContent = scene.title;
    $('scene-coordinates').textContent = coordinates(scene);
    $('scene-type').textContent = scene.type === 'landmark' ? 'Landmark' : 'Non-landmark';
    document.querySelectorAll('.scene-button').forEach((button, i) => button.setAttribute('aria-pressed', String(i === state.scene)));
    document.querySelectorAll('[data-gender]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.gender === state.gender)));
    document.querySelectorAll('[data-attire]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.attire === state.attire)));
    const items = [{ label: 'Original background', image: scene.original, original: true }, ...groups.map((group) => ({
      label: group,
      image: scene.variants.find((variant) => variant.group === group && variant.gender === state.gender && variant.attire === state.attire).image,
      original: false
    }))];
    const fragment = document.createDocumentFragment();
    items.forEach((item) => {
      const figure = document.createElement('figure');
      figure.className = `example-card${item.original ? ' original' : ''}`;
      if (!item.original) figure.style.setProperty('--group-color', groupColors[item.label]);
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'image-button';
      const attireName = state.attire === 'Casual' ? 'Unconditioned' : 'Race-associated';
      const alt = item.original ? `Original background photograph of ${scene.title}.` : `Counterfactual dataset edit of ${scene.title}: inserted ${state.gender.toLowerCase()} subject, ${item.label} annotation, ${attireName.toLowerCase()} attire. The original GPS label is retained.`;
      const caption = item.original ? `${scene.title} · Original background · ${coordinates(scene)}` : `${scene.title} · ${item.label} · ${state.gender} · ${attireName} attire · ${coordinates(scene)}`;
      button.setAttribute('aria-label', `Inspect ${item.original ? 'original background' : `${item.label} variant`} from ${scene.title}`);
      const image = document.createElement('img');
      image.src = item.image;
      image.alt = alt;
      image.loading = 'lazy';
      image.decoding = 'async';
      image.width = 1000;
      image.height = 667;
      const zoom = document.createElement('span');
      zoom.className = 'zoom-icon';
      zoom.textContent = '↗';
      zoom.setAttribute('aria-hidden', 'true');
      button.append(image, zoom);
      button.addEventListener('click', () => openImage(item.image, caption, alt));
      const figcaption = document.createElement('figcaption');
      const label = document.createElement('span');
      label.className = 'card-label';
      const dot = document.createElement('span');
      dot.className = 'card-dot';
      dot.setAttribute('aria-hidden', 'true');
      label.append(dot, document.createTextNode(item.original ? 'Original scene' : item.label));
      const detail = document.createElement('span');
      detail.className = 'card-detail';
      detail.textContent = item.original ? 'Fixed GPS label' : `${state.gender} · ${attireName}`;
      figcaption.append(label, detail);
      figure.append(button, figcaption);
      fragment.append(figure);
    });
    $('example-grid').replaceChildren(fragment);
    $('gallery-status').textContent = `${scene.title}. Original scene and seven ${state.gender.toLowerCase()} variants with ${state.attire === 'Casual' ? 'unconditioned' : 'race-associated'} attire. All retain the original ground-truth coordinates.`;
    renderComparison();
  }
  document.querySelectorAll('[data-gender]').forEach((button) => button.addEventListener('click', () => { state.gender = button.dataset.gender; renderGallery(); }));
  document.querySelectorAll('[data-attire]').forEach((button) => button.addEventListener('click', () => { state.attire = button.dataset.attire; renderGallery(); }));
  renderGallery();

  content.results.forEach((result) => {
    const row = document.createElement('div');
    row.className = 'chart-row';
    const name = document.createElement('span');
    name.className = 'model-name';
    name.textContent = result.model;
    const pair = document.createElement('div');
    pair.className = 'bar-pair';
    ['original', 'edited'].forEach((condition) => {
      const bar = document.createElement('div');
      bar.className = `bar ${condition}`;
      bar.style.setProperty('--value', result[condition]);
      bar.setAttribute('role', 'img');
      bar.setAttribute('aria-label', `${result.model}, ${condition === 'original' ? 'original background' : 'with inserted subject'}: ${result[condition].toFixed(1)} percent accuracy`);
      const value = document.createElement('span');
      value.textContent = result[condition].toFixed(1);
      bar.append(value);
      pair.append(bar);
    });
    row.append(name, pair);
    $('results-bars').append(row);
    const metrics = document.createElement('tr');
    [result.model, result.attack, result.help, result.distraction].forEach((value, index) => {
      const cell = document.createElement(index === 0 ? 'th' : 'td');
      if (index === 0) cell.scope = 'row';
      cell.textContent = index === 0 ? value : value.toFixed(1);
      metrics.append(cell);
    });
    $('popi-results').append(metrics);
  });

  const copyButton = $('copy-bibtex');
  copyButton.addEventListener('click', async () => {
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(content.citation);
        copied = true;
      }
    } catch { /* An available Clipboard API may still be denied. Use the legacy fallback. */ }
    if (!copied) {
      const field = document.createElement('textarea');
      field.value = content.citation;
      field.style.cssText = 'position:fixed;left:-9999px;top:0';
      field.setAttribute('aria-hidden', 'true');
      document.body.append(field);
      field.select();
      try { copied = document.execCommand('copy'); } catch { copied = false; }
      field.remove();
      copyButton.focus();
    }
    if (copied) {
      copyButton.querySelector('span').textContent = 'Copied!';
      $('copy-status').textContent = 'BibTeX citation copied to the clipboard.';
    } else {
      const range = document.createRange();
      range.selectNodeContents($('bibtex-code'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      $('copy-status').textContent = 'The citation is selected. Press Ctrl+C (or ⌘C) to copy, or download the .bib file.';
    }
  });

  if ('IntersectionObserver' in window) {
    const links = [...document.querySelectorAll('nav a')];
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) links.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`));
      });
    }, { rootMargin: '-15% 0px -60% 0px' });
    links.forEach((link) => observer.observe(document.querySelector(link.getAttribute('href'))));
  }
})();
