/**
 * Keyboard-driven animation-time playback control.
 *
 * A key press toggles playback, letting the frame loop gate the
 * simulation passes on {@link TimeControls.isPlaying} while the camera
 * stays interactive regardless of playback state.
 *
 * The controls are passive — they only maintain a boolean flag. The
 * caller is responsible for reading {@link isPlaying} each frame.
 */

/** Physical key code that toggles playback. */
const TOGGLE_KEY = "KeyP";

/**
 * Maintains animation-time playback state and toggles it from keyboard
 * events on the window.
 *
 * Modifier combinations and auto-repeats are ignored so ordinary text
 * entry and shortcuts are unaffected. Playback starts paused. Call
 * {@link dispose} to remove the listener.
 */
export class TimeControls {
    private playing: boolean = true;

    private readonly onKeyDown: (e: KeyboardEvent) => void;

    /**
     * Creates the controls and attaches the key listener to the window.
     */
    public constructor() {
        this.onKeyDown = (e: KeyboardEvent): void => {
            if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) {
                return;
            }
            if (e.code === TOGGLE_KEY) {
                this.playing = !this.playing;
            }
        };

        window.addEventListener("keydown", this.onKeyDown);
    }

    /**
     * Whether animation time currently advances.
     */
    public get isPlaying(): boolean {
        return this.playing;
    }

    /**
     * Removes the event listener from the window.
     *
     * Call this when the controls are no longer needed to avoid
     * dangling references.
     */
    public dispose(): void {
        window.removeEventListener("keydown", this.onKeyDown);
    }
}
