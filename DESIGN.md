# AnimaLoraStudio Design System

This document is the durable visual contract for Studio Web. Product behavior and
information architecture remain authoritative in the application and product docs;
this file defines how those behaviors are presented consistently.

## 1. Direction

AnimaLoraStudio is a focused creative workbench, not a marketing surface. Its visual
world is warm ivory, restrained orange, precise typography, and compact technical
controls. Light and dark themes express the same hierarchy. Changes should evolve
this world rather than replace it.

The interface serves two audiences at once:

- New users need clear hierarchy, familiar controls, and visible next actions.
- Experienced users need dense parameter editing, fast scanning, and stable layouts.

Consistency means equal semantics receive equal treatment. It does not mean every
surface has identical density.

## 2. Sources of truth

| Layer | Authority | Responsibility |
| --- | --- | --- |
| Foundation | `studio/web/src/styles/tokens.css` | Color, type, spacing, radius, shadow, motion, control states |
| Utility bridge | `studio/web/tailwind.config.js` | Maps CSS tokens into Tailwind utilities |
| Primitives | `studio/web/src/components/Button.tsx`, `Badge.tsx`, `Card.tsx`, `EmptyState.tsx`, `FormControl.tsx`, `Alert.tsx` | Typed, accessible component APIs |
| Patterns | `PageHeader`, `StepShell`, `ActionGroup`, `SaveIndicator`, `SaveBar`, `ListToolbar`, `Dialog`, `Modal`, `Drawer`, `Tabs`, `SegmentedControl`, `Toast`, `Field` | Repeated page and interaction structures |
| Product surfaces | `studio/web/src/pages/` | Business state and composition, not new visual primitives |

A page may compose primitives with layout utilities. It must not recreate an
existing primitive with arbitrary colors, padding, font sizes, or hover states.

## 3. Foundations

### Color

Use semantic tokens instead of literal colors:

- `canvas` is the page field.
- `surface` is the normal content plane.
- `sunken` is for wells, navigation, and code/data regions.
- `elevated` is for overlays and popovers.
- `accent` identifies the primary action or active process.
- `ok`, `warn`, `err`, and `info` communicate state, never decoration alone.

Do not rely on color as the only state cue. Pair it with text, an icon, or an
indicator. Dark mode must preserve hierarchy rather than simply invert colors.

### Typography

The Studio uses the local system sans stack for interface copy and the local mono
stack for machine-readable data. Do not add a network font dependency: the app
must remain readable while offline and should not shift when a font finishes
loading. Use no more than `400`, `500`, and `600` for normal UI hierarchy.

| Role | Contract | Use |
| --- | --- | --- |
| Page title | `.type-page-title`: `text-2xl`, 600, primary | One `h1` for the current page or workflow step |
| Page description | `.type-page-description`: `text-md`, secondary, relaxed, max `68ch` | A concise explanation directly below the page title |
| Section title | `.type-section-title`: `text-lg`, 600, primary | A major region inside a page or dialog |
| Panel title | `.type-panel-title`: `text-sm`, 600, primary | A card, settings group, or compact panel |
| Section label | `.type-section-label`: `text-xs`, 600, tertiary, tracked uppercase | A direct category heading such as queue status; never an eyebrow above another heading |
| Field label | `.type-field-label`: `text-sm`, 500, secondary | The human-readable name of a form control |
| Field help | `.type-field-help`: `text-xs`, tertiary, relaxed | Optional supporting copy below a field |
| Metadata | `text-xs` + tertiary | Timestamps, counts, and passive context |
| Technical data | `font-mono`; add `.tnum` for comparable numbers | Code, paths, identifiers, logs, and measurements—not generic UI chrome |

Rules:

- Normal interface and body copy uses `text-base`; compact controls and secondary
  content use `text-sm`. `text-2xs` is reserved for dense supporting labels and
  must never carry a primary action or required instruction.
- Keep normal reading copy between `65ch` and `75ch`; headings and control labels
  remain content-sized rather than stretching across the viewport.
- Choose heading elements by document structure, then apply the matching role.
  Do not skip levels to obtain a visual size.
- Use sentence case for ordinary headings. Uppercase/tracking is reserved for a
  category label that stands on its own; it is not a decorative kicker.
- The configured density changes the scale without changing semantic roles.
- Arbitrary pixel font sizes require a documented layout constraint and should
  remain exceptional.

### Spacing, radius, and depth

Use the `--s-*`, `--r-*`, and `--sh-*` scales. New shared layouts use the
semantic Tailwind spacing aliases below instead of choosing a number by eye:

| Alias | Default | Relationship |
| --- | ---: | --- |
| `related` | 8px | Icons, labels, and controls that form one action or datum |
| `field` | 12px | Parts of one field or compact component |
| `section` | 16px | Sibling groups within one region |
| `page-start` | 20px | Page-header leading inset |
| `page` | 24px | Page-shell inset and separation between ordinary regions |
| `page-loose` | 32px | Major region separation where a stronger pause is required |

The aliases resolve through `--space-*` to the density-aware `--s-*` scale.
Small `related` gaps stay stable; `field` and larger relationships contract or
expand with the selected density. Numeric spacing utilities remain a compatibility
path and migrate by component family. Do not globally remap them in a page PR.

Vertical rhythm follows content hierarchy: elements that form one control stay
closest, fields form a tighter group than sections, and sections form a tighter
group than page regions. A heading has more separation from the preceding region
than from the content it introduces. Shared page chrome uses semantic spacing so
its header, optional toolbar, and content keep aligned horizontal insets.

Within one hierarchy level, use one radius: controls use `--r-md`, ordinary cards
use `--r-lg`, and pills use `--r-pill`. Dense workbench panels may use `--r-md`
when their compactness is part of the information structure. Fixed or arbitrary
spacing is allowed only where it is geometry rather than rhythm—for example canvas
coordinates, image crops, table column sizing, or sticky offsets—and must remain
local to that specialized surface.

Shadows communicate elevation. Borders communicate grouping. Do not add shadows
merely to decorate every container.

### Density

- **Default** is the baseline for navigation, settings, dialogs, and ordinary forms.
- **Compact** is allowed for parameter-heavy workbenches, tables, and repeated row
  controls. It must use an explicit small component size or the global tight density,
  not local pixel values.
- **Loose** increases readability without changing component semantics.

## 4. Button contract

Use `Button` for button elements. Links that navigate may use `buttonClassName()`
when they need button presentation while retaining link semantics.

| Variant | Use | Do not use for |
| --- | --- | --- |
| `primary` | The single leading action in a local decision scope | Every positive action on a page |
| `secondary` | Normal actions and alternate choices | Passive navigation or icon-only chrome |
| `ghost` | Low-emphasis actions, toolbar controls, dismissals | Destructive actions without another cue |
| `warning` | Interrupting or canceling reversible/in-progress work | Permanent deletion |
| `danger` | Irreversible deletion or discarding recoverable state | Routine cancellation |

Sizes:

- `md`: ordinary forms and dialogs.
- `sm`: headers, cards, and compact action rows.
- `xs`: dense tables, filters, and micro toolbars; never body copy squeezed smaller.

Rules:

- Button labels name the action.
- Icon-only buttons require an accessible label.
- Loading buttons remain labeled, expose `aria-busy`, and cannot be activated.
- The imperative `useDialog()` API preserves the product's single-accent confirmation
  convention: its `tone` marks urgent dialog semantics but does not recolor the confirm
  button. Declarative destructive action groups continue to use `danger`.
- Toggle buttons expose `aria-pressed`.
- Disabled, hover, active, and keyboard-focus states come from the primitive.
- Do not combine `bg-*`, `border-*`, and `text-*` to reinvent an existing variant.

## 5. Badge contract

Badges are non-interactive labels. A clickable pill is a button or link, not a badge.

| Tone | Meaning | Common states |
| --- | --- | --- |
| `neutral` | Passive, queued, unknown, or canceled metadata | pending, scheduled, canceled |
| `accent` | Active work or the current process | running, training, evaluating |
| `success` | Successful completion or confirmed availability | done, completed, available |
| `warning` | Paused, partial, or attention required | paused, partial |
| `danger` | Failure or invalid state | failed, error |
| `info` | Informational classification without success/failure | source or category labels |

An active badge may include the shared pulsing indicator. Use `sm` only for dense
metadata such as announcement tags; status badges use the default size.

Domain components such as `VersionStatusBadge` own the mapping from backend state to
badge tone. The generic `Badge` component must not know API enums.

## 6. Surface and empty-state contract

Use `Card` for ordinary bordered content surfaces. The default card uses the ordinary
`--r-lg` radius; compact workbench panels opt into the compact radius explicitly.
Padding belongs to the component API when it describes the whole surface, so density
modes can adjust it through spacing tokens. Interactive links and buttons retain their
native semantics and use `cardClassName()` for card presentation.

Use `EmptyState` when a list or page region has no content to present:

- `md` is the primary zero state for an otherwise empty page or tab.
- `sm` is a compact no-match state inside an existing section.
- Titles state what is absent; descriptions explain the next useful step.
- Actions are optional and must use the appropriate Button or link primitive.
- Loading, errors, and warnings are not empty states and require their own semantics.

Do not reproduce `rounded-* + border-subtle + bg-surface` for an ordinary card or
hand-build centered zero-state typography on product pages.

## 7. Form-control contract

Use the typed `Input`, `Select`, `Textarea`, and `Checkbox` primitives for ordinary
native form controls. They preserve browser semantics while sharing focus, disabled,
invalid, sizing, and theme behavior. The legacy `.input` and `.input-mono` classes
remain compatibility paths during incremental migration.

Control sizes:

- `md` is the default for dialogs and ordinary forms and aligns with `Button md`.
- `sm` is for settings rows, filter toolbars, schema forms, and other explicitly
  compact workbench regions; it must not be recreated with local padding overrides.

Control surfaces communicate placement, not state:

- `surface` is the ordinary form plane and default.
- `canvas` is the compact Schema `Field` treatment inside a surrounding surface.
- `sunken` is for settings wells and data-entry regions designed as inset controls.

Use `mono` only for paths, identifiers, JSON, numeric technical values, and other
machine-readable content. Errors pair visible recovery copy with `invalid`, which
exposes `aria-invalid` and the shared error border/focus ring. Disabled appearance,
keyboard focus, placeholder color, and checkbox accent come from the primitive.
Do not replace a native select or checkbox with a custom interaction only to alter
its appearance.

The primitive owns presentation only. Debouncing, parsing, commit-on-blur, schema
validation, picker composition, and business state remain in Field or product
patterns. File/color/range inputs and composite pickers require their own contracts
and are not styled as text controls by default.

Settings explanations belong in the label-adjacent `InfoButton` tooltip according
to `docs/design/ui-info-design.md`; do not add permanent explanatory paragraphs
under individual settings.

## 8. Feedback contract

Use `Alert` for persistent, in-flow information, success confirmation, warnings, and
errors. `Toast` remains the transient notification pattern and reuses Alert's visual
tones without changing its timer or invocation API.

| Tone | Meaning | Typical use |
| --- | --- | --- |
| `info` | Neutral operational context or guidance | non-blocking system information |
| `success` | A completed action that remains relevant in the current view | saved or imported confirmation |
| `warning` | Attention or recovery is required, but the state is not yet a failure | paused or held work |
| `danger` | An operation failed or content is invalid | failed loads and rejected operations |

Use `md` for ordinary page feedback and `sm` inside compact workbench regions. A
semantic icon accompanies each tone so meaning does not depend on color alone. Titles
are optional and should name the condition; body copy explains the consequence or
recovery. Actions belong in the dedicated action slot and use `Button`.

ARIA live behavior is explicit because not every visible notice is newly announced:
use `role="alert"` only for urgent dynamic failures, `role="status"` for non-urgent
dynamic confirmation, and no live role for persistent page context. Do not announce the
same event through both an in-flow Alert and a Toast; choose the surface nearest to the
recovery action. Field validation stays adjacent to its control; empty states, domain
status cards, and modal workflow steps are not Alerts. Error copy and recovery details
must wrap rather than truncate.

Do not recreate feedback with local `bg-*-soft + border-* + text-*` combinations.

## 9. Action-area and save-pattern contract

`ActionGroup` is the typed Pattern-layer entry point for related save, submit, and
recovery controls. Its slots render in a stable order: optional status first,
secondary or destructive actions next, and the single primary action last. The primary
action therefore stays at the far right in left-to-right layouts. A visual divider may
separate destructive or context-changing actions, but it does not create another
primary action.

Save and submit buttons are text-first. Do not prefix ordinary labels with floppy-disk,
checkmark, or other decorative emoji; reserve icons for established compact utilities,
and provide an accessible label for icon-only controls. Use `Button` loading and disabled
states rather than replacing the label with an unrelated spinner. An explicit save may
lose primary emphasis when nothing is dirty, but it remains in the primary slot so the
layout does not jump.

Placement follows editing scope:

- Use the page or step header for short, viewport-contained edits and quick actions.
- Use a footer action area for long or scroll-heavy forms that require deliberate review
  before submit. Make it sticky only when the final action would otherwise be difficult
  to reach, and reserve content space so it never obscures fields.
- Keep actions that affect only one panel inside that panel; do not promote them to a
  page-level bar.
- Autosave surfaces show `SaveIndicator` status instead of a redundant Save button.
  If an error Toast already announces the same failure, disable the indicator's error
  announcement so assistive technology receives it only once.

Status copy precedes controls and uses a stable polite live region. At narrow widths,
action groups may wrap, but the primary action remains last and right-aligned. Snapshot
restore, reset, and destructive actions remain secondary to saving unless that recovery
operation is the sole purpose of the current scope.

## 10. Modal-dialog contract

`Modal` is the declarative Pattern-layer shell for interruptive tasks that require
protected focus: confirmations, short forms, option selection, and structured review.
Use the imperative `useDialog()` API for simple text-only alert, confirm, and prompt
flows; it renders through the same shell. Use a Drawer for persistent secondary work
that should remain available alongside page context, and do not put ordinary page
content in a modal merely to make it prominent.

Every modal has one required title, an optional concise description, one scrollable body,
and an optional footer. Use `sm` for a short prompt, `md` for ordinary forms, and `lg`
for structured comparisons or dense option sets. The panel is portalled above the app,
keeps viewport-safe outer padding and a bounded height, and leaves the title and footer
visible while long body content scrolls.

Dismissal and focus are part of the Pattern rather than caller-owned behavior:

- Opening moves focus into the dialog; closing restores focus to the opener.
- Tab and Shift+Tab remain within the active dialog, and Escape dismisses unless an
  irreversible or in-progress operation explicitly disables it.
- A backdrop press may dismiss a reversible dialog. Pressing inside the panel, or
  starting an interaction inside and releasing outside, must not dismiss it.
- The shell supplies `role="dialog"`, `aria-modal`, and title/description linkage. Use
  `alertdialog` only when an immediate decision is required before work can continue.
- Lock background scrolling while open. Do not stack modal dialogs.

Footer actions use `ActionGroup`: status first, secondary or destructive actions next,
and the single primary action last. Keep validation near the relevant control; an error
Toast must not duplicate an error already announced inside the modal. Toast feedback remains
above the modal layer when an operation keeps the dialog open. Do not recreate modal
backdrops, panel geometry, focus listeners, or title linkage in feature code.

## 11. Overlay-drawer contract

Use `Drawer` for an interruptive task that slides above the current workspace while
preserving visible context. It is an overlay side sheet, not a generic name for every
panel attached to an edge: bottom task logs remain page-owned footer panels, and Generate
pickers/editors remain attached workspaces with their own keep-alive geometry.

Drawer motion has one owner and one lifecycle: `closed → opening → open → closing`.
The shell remains mounted so cold and warm opens take the same path. Feature content must
not add a second entrance animation or independently decide when the panel becomes visible.
Static local content mounts in the same commit as the shell and moves with the panel; once
mounted, keep it alive across later opens. Do not insert a transient loading label or skeleton
for a page whose code and structure are already local—it creates flicker without communicating
real progress. Use a local skeleton only for genuinely asynchronous remote content with a
perceptible wait, and never replace the whole Drawer shell. Reduced-motion skips the authored
transition consistently.

Geometry and interaction belong to the Pattern:

- The panel is portalled above the AppShell and below Modal/Toast layers. Its width is always
  bounded by the viewport; long content scrolls inside the panel and never widens the page.
- The backdrop and panel animate as one authored moment. Closing is shorter than opening;
  loading, section changes, and lazy-module timing must not alter the shell motion.
- Opening focuses the panel or an explicit initial target. Tab remains inside, Escape and a
  direct backdrop press request a reversible close, and closing restores focus to the opener.
- While open, the application root is inert. Do not mutate `body` overflow for Studio drawers:
  the AppShell owns its own scroll container, and body scrollbar changes cause layout shift.
- A Modal opened from a Drawer is the active top layer. Escape closes the Modal first; the
  Drawer remains until its own close request succeeds.
- Deep links or `open({ section })` requests wait for `open` readiness, then scroll only the
  Drawer content owner. Never call early `scrollIntoView` in parallel with panel motion.

Settings is the first representative adoption. Its data and instant-save behavior remain
outside the Drawer Pattern; only shell timing, focus, dismissal, width, and internal scroll
readiness are shared.

## 12. Tabs and segmented-selection contract

Use `Tabs` for navigation among peer content panels and `SegmentedControl` for changing
one mutually exclusive value or working mode. They share visual rhythm and keyboard
behavior, but their semantics are not interchangeable: Tabs render `tablist` / `tab`
and reference labelled `tabpanel` content; segmented values render `radiogroup` /
`radio`. Do not use tab semantics for filters, view toggles, or actions that navigate to
a different route.

Tabs support two appearances without changing meaning:

- `underline` is the default for page-width or panel-level content sections. At narrow
  widths it scrolls horizontally instead of wrapping labels into ambiguous rows.
- `segmented` is for compact, bounded navigation inside a workbench panel. Use the same
  segmented appearance for mutually exclusive modes, but through `SegmentedControl`.

Selection follows focus for this application: Arrow keys move to the next enabled item
and activate it, wrapping at both ends; Home and End move to the first and last enabled
item. Exactly one enabled item participates in the page Tab order. Disabled options are
skipped. Every group has an accessible label, and every content tab supplies stable
`aria-controls` / `aria-labelledby` linkage. Do not recreate local active-state class
strings or implement feature-owned roving focus.

Use `sm` for dense nested controls such as X/Y axes and sidebar sections; use `md` for
primary panel navigation and mode choice. Compact selection labels may use `text-2xs`
because they are peer navigation, not a primary action or required instruction. Labels
may remain on one line in a segmented track only when the full label remains available
as the accessible name and title. Underlined tabs keep their intrinsic width and scroll
rather than truncating. Both appearances must preserve visible focus, disabled state,
theme contrast, density response, and Chinese/English label stability.

## 13. List-toolbar contract

Use `ListToolbar` for an ordinary list's collapsible search, facet-filter, and sort
region immediately below its page header. It is not the page action area, a bulk
mutation bar, or a professional workbench toolbar. `PageHeader` continues to own the
disclosure and refresh actions; Gallery autocomplete, draft filter popovers, Eval axis
selection, and tag bulk editing retain their domain-specific composition.

The Pattern owns a named region and stable slot order: dominant search first, filters
second, and sort last. This is also the DOM and keyboard order. At wide desktop widths,
the query owns the flexible primary track while compact facets remain trailing. At the
project's approved narrow-desktop breakpoint, the query takes a full first row and the
trailing controls wrap below it without changing order. Page-aligned padding, borders,
gaps, theme colors, and density response come from semantic tokens; callers must not
restore fixed percentage widths.

The disclosure button supplies `aria-expanded` and `aria-controls`. Keep the current
region mounted with `hidden` while collapsed so the ID reference remains valid and the
row takes no layout space. Every region has a localized accessible name, and each input
or select retains its own label. The Pattern is presentational: query debounce,
persistence, active-filter dots, sorting, API parameters, paging resets, result counts,
refresh, clear behavior, and list mutations remain page-owned. Do not add unused result
or clear slots until a repeated product behavior has been established.

## 14. Accessibility and resilience

Every primitive must work with keyboard focus, disabled state, light/dark themes,
all three density modes, and both Chinese and English labels. Focus is always visible.
Controls must tolerate longer English copy without fixed-width truncation unless the
full value is available through an established disclosure pattern.

Respect `prefers-reduced-motion`; status information must remain understandable when
animation is disabled.

## 15. Migration policy

Migration is incremental:

1. Preserve the CSS classes as compatibility primitives.
2. Use typed components for new work.
3. Migrate representative surfaces with tests.
4. Move remaining pages by component family, not by arbitrary page batches.
5. Remove old compatibility paths only after repository-wide adoption.

A migration must preserve business behavior, route contracts, schema, SSE events,
and localization unless those changes are explicitly part of its scope.
