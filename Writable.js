// store.js
import { writable } from 'svelte/store';

export const feature_names = writable([]); // Export a reactive array
export const hidden_nodes = writable([]); // Export a reactive array
export const features = writable([]);
export const outputs = writable([]);