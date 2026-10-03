const tabs = [...document.querySelectorAll('[data-tab]')];
const panels = [...document.querySelectorAll('[data-panel]')];

function selectDataset(name, focus = false) {
  for (const tab of tabs) {
    const selected = tab.dataset.tab === name;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  }
  for (const panel of panels) {
    panel.hidden = panel.dataset.panel !== name;
  }
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectDataset(tab.dataset.tab));
  tab.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    const next = (index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length;
    selectDataset(tabs[next].dataset.tab, true);
  });
});

const dialog = document.getElementById('image-dialog');
const dialogImage = document.getElementById('dialog-image');
document.querySelectorAll('[data-zoom]').forEach((button) => {
  button.addEventListener('click', () => {
    dialogImage.src = button.dataset.zoom;
    dialogImage.alt = button.dataset.alt;
    dialog.showModal();
  });
});
document.getElementById('close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close();
});
