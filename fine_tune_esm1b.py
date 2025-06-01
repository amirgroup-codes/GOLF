#!/usr/bin/env python3
"""
Fine-tune ESM-1b model on Multiple Sequence Alignment (MSA) data.
The script freezes the early layers of the model and only fine-tunes
the last few layers for efficiency and to prevent overfitting.
Supports both FASTA and A3M formats for MSA input.
"""

import os
import argparse
import logging
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import re
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import esm

# Set up logging with more detailed format including file and line number
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

class ProteinMSADataset(Dataset):
    """
    Dataset for loading protein MSA data and optionally labels for supervised fine-tuning
    """
    def __init__(self, msa_file, alphabet, max_seqs=None, max_length=1024, labels_file=None, sample_method='random', msa_format='auto'):
        """
        Args:
            msa_file (str): Path to MSA file in a3m or FASTA format
            alphabet: ESM alphabet for tokenization
            max_seqs (int): Maximum number of sequences to use (None=use all)
            max_length (int): Maximum sequence length to keep
            labels_file (str): Optional path to labels file for supervised learning
            sample_method (str): How to sample sequences ('random', 'first', or 'evenly')
            msa_format (str): Format of MSA file ('auto', 'a3m', or 'fasta')
        """
        logger.info(f"Initializing ProteinMSADataset with file: {msa_file}")
        logger.info(f"Parameters: max_seqs={max_seqs}, max_length={max_length}, sample_method={sample_method}")
        
        self.msa_file = msa_file
        self.alphabet = alphabet
        self.max_length = max_length
        self.sample_method = sample_method
        self.msa_format = msa_format
        
        # Determine format from file extension if auto
        if self.msa_format == 'auto':
            if msa_file.lower().endswith('.a3m'):
                self.msa_format = 'a3m'
            elif any(msa_file.lower().endswith(ext) for ext in ['.fasta', '.fa', '.fna', '.faa']):
                self.msa_format = 'fasta'
            else:
                # Default to a3m for files without recognized extension
                logger.warning(f"Could not determine MSA format from file extension. Assuming A3M format for {msa_file}")
                self.msa_format = 'a3m'
                
        logger.info(f"Using MSA format: {self.msa_format}")
        
        # Load MSA data
        logger.info(f"Loading MSA from {msa_file}")
        start_time = time.time()
        self.seqs = self._read_msa(msa_file)
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(self.seqs)} sequences from MSA file in {load_time:.2f} seconds")
        
        # Print a sample of the first sequence for verification
        if self.seqs:
            first_seq = self.seqs[0]
            logger.info(f"First sequence ID: {first_seq['id']}")
            seq_preview = first_seq['seq'][:50] + '...' if len(first_seq['seq']) > 50 else first_seq['seq']
            logger.info(f"First sequence preview: {seq_preview}")
            logger.info(f"First sequence length: {len(first_seq['seq'])}")
        else:
            logger.error("No sequences were loaded from the MSA file!")
        
        # Sample sequences if needed
        if max_seqs is not None and max_seqs < len(self.seqs):
            logger.info(f"Sampling {max_seqs} sequences using method: {sample_method}")
            start_time = time.time()
            self.seqs = self._sample_seqs(self.seqs, max_seqs)
            sample_time = time.time() - start_time
            logger.info(f"Sampled {len(self.seqs)} sequences in {sample_time:.2f} seconds")
        
        # Load labels if provided
        self.labels = None
        if labels_file is not None and os.path.exists(labels_file):
            logger.info(f"Loading labels from {labels_file}")
            self.labels = pd.read_csv(labels_file)
            # Map labels to sequence IDs here
            logger.info(f"Loaded {len(self.labels)} labels")
    
    def _read_msa(self, msa_file):
        """Read sequences from an MSA file in A3M or FASTA format"""
        if self.msa_format == 'a3m':
            return self._read_a3m(msa_file)
        elif self.msa_format == 'fasta':
            return self._read_fasta(msa_file)
        else:
            raise ValueError(f"Unknown MSA format: {self.msa_format}")
            
    def _read_a3m(self, msa_file):
        """Read sequences from an A3M format file"""
        logger.info(f"Reading A3M format file: {msa_file}")
        sequences = []
        
        try:
            current_seq = {"id": None, "seq": ""}
            line_count = 0
            seq_count = 0
            lowercase_detected = False
            
            with open(msa_file, 'r') as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq["id"] is not None:
                            if len(current_seq["seq"]) <= self.max_length:
                                # Check for lowercase characters
                                if any(c.islower() for c in current_seq["seq"]):
                                    lowercase_detected = True
                                sequences.append(current_seq)
                                seq_count += 1
                            else:
                                logger.debug(f"Skipped sequence {current_seq['id']} as it exceeds max_length ({len(current_seq['seq'])} > {self.max_length})")
                        current_seq = {"id": line[1:], "seq": ""}
                    else:
                        # Remove lowercase characters (insertions in A3M format)
                        current_seq["seq"] += ''.join([c for c in line if c.isupper() or c == '-'])
            
            # Add the last sequence
            if current_seq["id"] is not None and len(current_seq["seq"]) <= self.max_length:
                # Check for lowercase characters
                if any(c.islower() for c in current_seq["seq"]):
                    lowercase_detected = True
                sequences.append(current_seq)
                seq_count += 1
            elif current_seq["id"] is not None:
                logger.debug(f"Skipped last sequence {current_seq['id']} as it exceeds max_length")
            
            logger.info(f"Read {line_count} lines, found {seq_count} valid sequences in A3M file")
            
            if lowercase_detected:
                logger.warning("Lowercase characters detected in sequences. These will be converted to uppercase during processing.")
            
        except Exception as e:
            logger.error(f"Error reading A3M file: {e}")
            raise
            
        return sequences
    
    def _read_fasta(self, msa_file):
        """Read sequences from a FASTA format file"""
        logger.info(f"Reading FASTA format file: {msa_file}")
        sequences = []
        
        try:
            current_seq = {"id": None, "seq": ""}
            line_count = 0
            seq_count = 0
            lowercase_detected = False
            
            with open(msa_file, 'r') as f:
                for line in f:
                    line_count += 1
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq["id"] is not None:
                            if len(current_seq["seq"]) <= self.max_length:
                                # Check for lowercase characters
                                if any(c.islower() for c in current_seq["seq"]):
                                    lowercase_detected = True
                                sequences.append(current_seq)
                                seq_count += 1
                            else:
                                logger.debug(f"Skipped sequence {current_seq['id']} as it exceeds max_length ({len(current_seq['seq'])} > {self.max_length})")
                        current_seq = {"id": line[1:].split()[0], "seq": ""}  # Extract ID (up to first space)
                    elif line and not line.startswith(';'):  # Skip comment lines
                        # In FASTA, all characters are part of the sequence
                        current_seq["seq"] += line.replace(" ", "")
            
            # Add the last sequence
            if current_seq["id"] is not None and len(current_seq["seq"]) <= self.max_length:
                # Check for lowercase characters
                if any(c.islower() for c in current_seq["seq"]):
                    lowercase_detected = True
                sequences.append(current_seq)
                seq_count += 1
            elif current_seq["id"] is not None:
                logger.debug(f"Skipped last sequence {current_seq['id']} as it exceeds max_length")
            
            logger.info(f"Read {line_count} lines, found {seq_count} valid sequences in FASTA file")
            
            if lowercase_detected:
                logger.warning("Lowercase characters detected in sequences. These will be converted to uppercase during processing.")
            
            # Validate that all sequences have the same length for MSA
            if sequences:
                ref_length = len(sequences[0]["seq"])
                logger.info(f"Reference sequence length: {ref_length}")
                
                invalid_seqs = [i for i, s in enumerate(sequences) if len(s["seq"]) != ref_length]
                
                if invalid_seqs:
                    logger.warning(f"Found {len(invalid_seqs)} sequences with inconsistent lengths in FASTA MSA file.")
                    logger.warning(f"Reference length: {ref_length}, but found sequences with different lengths.")
                    if len(invalid_seqs) > 0:
                        sample_idx = invalid_seqs[0]
                        sample_seq = sequences[sample_idx]
                        logger.warning(f"Example inconsistent sequence: ID={sample_seq['id']}, length={len(sample_seq['seq'])}")
                    logger.warning("Filtering out sequences with inconsistent lengths.")
                    
                    sequences = [s for i, s in enumerate(sequences) if i not in invalid_seqs]
                    logger.info(f"Kept {len(sequences)} sequences with consistent length.")
                else:
                    logger.info(f"All {len(sequences)} sequences have consistent length of {ref_length}")
            
        except Exception as e:
            logger.error(f"Error reading FASTA file: {e}")
            raise
            
        return sequences
    
    def _sample_seqs(self, seqs, max_seqs):
        """
        Sample sequences from the MSA
        
        Args:
            seqs: List of sequence dictionaries
            max_seqs: Maximum number of sequences to sample
        
        Returns:
            Sampled sequences
        """
        if max_seqs >= len(seqs):
            logger.info(f"No sampling needed: requested {max_seqs} sequences, but only {len(seqs)} are available")
            return seqs
        
        # Always include the first sequence (wild-type)
        first_seq = seqs[0]
        remaining_seqs = seqs[1:]
        
        if self.sample_method == 'first':
            # Take the first max_seqs sequences
            sampled = remaining_seqs[:max_seqs-1]  # -1 to account for the first sequence we're keeping
        elif self.sample_method == 'evenly':
            # Sample evenly across the MSA
            indices = np.linspace(0, len(remaining_seqs) - 1, max_seqs - 1, dtype=int)
            sampled = [remaining_seqs[i] for i in indices]
        else:  # default to random
            # Random sampling
            sampled = random.sample(remaining_seqs, min(max_seqs - 1, len(remaining_seqs)))
        
        # Add the first sequence back at the beginning
        sampled = [first_seq] + sampled
        
        logger.info(f"Sampled {len(sampled)} sequences ({self.sample_method} sampling)")
        logger.info(f"First sequence (wild-type) ID: {first_seq['id']} is included in the sample")
        
        return sampled
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        """Get a sequence from the dataset"""
        if idx >= len(self.seqs):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.seqs)} sequences")
        
        sequence_data = self.seqs[idx]
        original_sequence = sequence_data["seq"]
        
        # Convert to uppercase to avoid tokenization issues
        sequence = original_sequence.upper()
        
        # Keep gaps since they have a token in the vocabulary (index 30)
        # No need to replace or remove them
        
        # Visualize sequence processing if in debug mode
        if logger.level <= logging.DEBUG and idx % 100 == 0:  # Only log every 100th sequence to avoid excessive logging
            visualize_sequence_processing(
                sequence_id=sequence_data["id"],
                original_sequence=original_sequence,
                processed_sequence=sequence
            )
        
        # Get label if available
        label = None
        if self.labels is not None and sequence_data["id"] in self.labels:
            label = self.labels[sequence_data["id"]]
        
        return sequence_data["id"], sequence, label

def mask_tokens(tokens, alphabet, mask_prob=0.15):
    """
    Mask tokens for masked language modeling (MLM) prediction
    
    Args:
        tokens: Tensor of token indices (batch_size, seq_len)
        alphabet: ESM alphabet object
        mask_prob: Probability of masking a token
    
    Returns:
        masked_tokens: Tensor with some tokens masked
        labels: Tensor with labels for MLM (-100 for non-masked tokens)
    """
    # Create masked tokens tensor (a copy of the original tokens)
    masked_tokens = tokens.clone()
    
    # Log the mask token index for debugging
    logger.debug(f"Using mask token index: {alphabet.mask_idx}")
    
    # Mask eligible tokens with mask_prob probability
    # Don't mask special tokens or gap characters
    special_tokens_mask = [
        alphabet.cls_idx,
        alphabet.eos_idx,
        alphabet.padding_idx,
        30  # Gap token index
    ]
    
    # Create labels tensor for MLM
    labels = torch.ones_like(tokens) * -100  # Initialize all to -100 (ignored in loss)
    
    # For each sequence in the batch
    for i in range(tokens.size(0)):
        # Identify eligible token positions (not special tokens or gaps)
        eligible_indices = []
        for j in range(tokens.size(1)):
            if tokens[i, j].item() not in special_tokens_mask:
                eligible_indices.append(j)
        
        # Randomly select positions to mask
        num_to_mask = int(len(eligible_indices) * mask_prob)
        mask_indices = random.sample(eligible_indices, min(num_to_mask, len(eligible_indices)))
        
        # Apply masking and set labels
        for j in mask_indices:
            # Store the original token as the label
            labels[i, j] = tokens[i, j].clone()
            # Replace with mask token
            masked_tokens[i, j] = alphabet.mask_idx
    
    # Log statistics about masking
    if logger.isEnabledFor(logging.DEBUG):
        total_masked = (labels != -100).sum().item()
        total_tokens = tokens.numel()
        logger.debug(f"Masked {total_masked} out of {total_tokens} tokens ({total_masked/total_tokens*100:.2f}% masked)")
    
    return masked_tokens, labels

def visualize_sequence_processing(sequence_id, original_sequence, processed_sequence, tokens=None, masked_tokens=None, alphabet=None):
    """
    Debug function to visualize sequence processing steps.
    
    Args:
        sequence_id: ID of the sequence
        original_sequence: Original sequence with potential gaps and lowercase
        processed_sequence: Processed sequence after removing gaps and converting to uppercase
        tokens: Token indices if available (after tokenization)
        masked_tokens: Masked token indices if available
        alphabet: ESM alphabet object for converting tokens back to amino acids
    """
    logger.debug(f"\n==== SEQUENCE VISUALIZATION: {sequence_id} ====")
    logger.debug(f"Original length: {len(original_sequence)}")
    logger.debug(f"Processed length: {len(processed_sequence)}")
    
    # Display the first 50 characters of the sequences
    preview_length = min(50, len(original_sequence), len(processed_sequence))
    logger.debug(f"Original seq (first {preview_length}): {original_sequence[:preview_length]}")
    logger.debug(f"Processed seq (first {preview_length}): {processed_sequence[:preview_length]}")
    
    # Show changes made to the sequence
    if len(original_sequence) != len(processed_sequence):
        logger.debug(f"Length changed: {len(original_sequence)} -> {len(processed_sequence)}")
    
    # Count gaps and lowercase characters in original
    gap_count = original_sequence.count('-')
    lowercase_count = sum(1 for c in original_sequence if c.islower())
    logger.debug(f"Gaps in sequence: {gap_count}, Lowercase chars converted: {lowercase_count}")
    
    # If tokens are provided, show token information
    if tokens is not None and alphabet is not None:
        logger.debug(f"Tokenized length: {len(tokens)}")
        
        # Convert tokens back to amino acids for the first few positions
        try:
            token_preview_length = min(20, len(tokens))
            token_preview = tokens[:token_preview_length].tolist()
            
            amino_acids = []
            for token in token_preview:
                if token == alphabet.cls_idx:
                    amino_acids.append("<cls>")
                elif token == alphabet.eos_idx:
                    amino_acids.append("<eos>")
                elif token == alphabet.padding_idx:
                    amino_acids.append("<pad>")
                elif token == alphabet.mask_idx:
                    amino_acids.append("<mask>")
                elif token == 30:  # Gap token
                    amino_acids.append("-")
                else:
                    # Try to get the amino acid for this token
                    for aa, idx in alphabet.tok_to_idx.items():
                        if idx == token:
                            amino_acids.append(aa)
                            break
                    else:
                        amino_acids.append(f"<{token}>")
            
            logger.debug(f"Token preview (first {token_preview_length}): {token_preview}")
            logger.debug(f"Amino acids from tokens: {''.join(amino_acids)}")
            
            # If masked tokens exist, show masking
            if masked_tokens is not None:
                masked_preview = masked_tokens[:token_preview_length].tolist()
                masked_positions = [i for i, (a, b) in enumerate(zip(token_preview, masked_preview)) if a != b]
                logger.debug(f"Masked positions: {masked_positions}")
                logger.debug(f"Masked tokens: {masked_preview}")
        except Exception as e:
            logger.error(f"Error visualizing tokens: {e}")
    
    logger.debug("="*40)

def visualize_batch(batch_ids, batch_sequences, batch_tokens=None, batch_masked=None, alphabet=None, max_examples=2):
    """
    Debug function to visualize a batch of sequences.
    
    Args:
        batch_ids: List of sequence IDs in the batch
        batch_sequences: List of sequences in the batch
        batch_tokens: Batch tokens tensor if available
        batch_masked: Masked batch tokens tensor if available
        alphabet: ESM alphabet object
        max_examples: Maximum number of examples to show
    """
    logger.debug("\n" + "="*20 + " BATCH VISUALIZATION " + "="*20)
    logger.debug(f"Batch size: {len(batch_sequences)}")
    
    # Show statistics about sequence lengths
    seq_lengths = [len(seq) for seq in batch_sequences]
    logger.debug(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={sum(seq_lengths)/len(seq_lengths):.1f}")
    
    # Count amino acid distribution
    aa_counts = {}
    gap_counts = []
    for seq in batch_sequences:
        gap_count = seq.count('-')
        gap_counts.append(gap_count)
        
        for aa in seq:
            if aa != '-':  # Count amino acids separately from gaps
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    # Log gap statistics
    if gap_counts:
        total_gaps = sum(gap_counts)
        avg_gaps = total_gaps / len(batch_sequences)
        logger.debug(f"Gap statistics: total={total_gaps}, avg={avg_gaps:.1f} per sequence, "
                   f"min={min(gap_counts)}, max={max(gap_counts)}")
    
    total_aas = sum(aa_counts.values())
    aa_distribution = {aa: f"{count/total_aas*100:.1f}%" for aa, count in sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)}
    logger.debug(f"Amino acid distribution: {aa_distribution}")
    
    # Show example sequences
    for i in range(min(max_examples, len(batch_sequences))):
        logger.debug(f"\nExample {i+1}:")
        logger.debug(f"ID: {batch_ids[i]}")
        preview_len = min(30, len(batch_sequences[i]))
        logger.debug(f"Sequence preview: {batch_sequences[i][:preview_len]}")
        logger.debug(f"Gaps in sequence: {batch_sequences[i].count('-')}")
        
        # Show token information if available
        if batch_tokens is not None and alphabet is not None:
            try:
                token_preview_length = min(20, batch_tokens.shape[1])
                example_tokens = batch_tokens[i, :token_preview_length].tolist()
                logger.debug(f"Tokens preview: {example_tokens}")
                
                # Count gap tokens
                gap_token_count = sum(1 for t in batch_tokens[i] if t.item() == 30)
                logger.debug(f"Gap tokens in tokenized sequence: {gap_token_count}")
                
                if batch_masked is not None:
                    example_masked = batch_masked[i, :token_preview_length].tolist()
                    mask_count = sum(1 for a, b in zip(example_tokens, example_masked) if a != b)
                    logger.debug(f"Masked tokens preview: {example_masked}")
                    logger.debug(f"Number of masked positions: {mask_count}")
            except Exception as e:
                logger.error(f"Error visualizing batch tokens: {e}")
    
    logger.debug("="*60)

def collate_batch(batch, alphabet, mask_prob=0.15):
    """
    Collate a batch of sequences for training
    
    Args:
        batch: List of (seq_id, sequence, label) tuples
        alphabet: ESM alphabet object
        mask_prob: Probability of masking a token
    
    Returns:
        inputs: Tensor of masked token indices
        labels: Tensor with labels for MLM (-100 for non-masked tokens)
    """
    # Unpack the batch
    seq_ids, sequences, labels = zip(*batch)
    
    # Log a sample from the batch for debugging
    if len(sequences) > 0:
        logger.debug(f"Sample sequence: {sequences[0][:50]}...")
    
    logger.debug(f"Converted all sequences to uppercase for tokenization")
    
    # Visualize the batch before tokenization
    if logger.level <= logging.DEBUG:
        visualize_batch(seq_ids, sequences)
    
    # Convert sequences to tokens
    batch_converter = alphabet.get_batch_converter()
    try:
        logger.debug(f"Converting {len(sequences)} sequences to tokens")
        batch_labels, batch_strs, batch_tokens = batch_converter(list(zip(seq_ids, sequences)))
        
        logger.debug(f"Tokenized batch shape: {batch_tokens.shape}")
        logger.debug(f"Token range: min={batch_tokens.min().item()}, max={batch_tokens.max().item()}")
        
        # Visualize tokens
        if logger.level <= logging.DEBUG and len(sequences) > 0:
            visualize_sequence_processing(
                sequence_id=seq_ids[0],
                original_sequence=sequences[0],
                processed_sequence=sequences[0],  # Already processed
                tokens=batch_tokens[0],
                alphabet=alphabet
            )
        
        # Apply masking for MLM
        masked_tokens, mlm_labels = mask_tokens(batch_tokens, alphabet, mask_prob)
        
        # Log masking statistics
        num_masked = (mlm_labels != -100).sum().item()
        total_tokens = batch_tokens.numel()
        logger.debug(f"Masked {num_masked} out of {total_tokens} tokens ({num_masked/total_tokens*100:.2f}%)")
        
        # Visualize masked batch
        if logger.level <= logging.DEBUG and len(sequences) > 0:
            visualize_sequence_processing(
                sequence_id=seq_ids[0],
                original_sequence=sequences[0],
                processed_sequence=sequences[0],  # Already processed
                tokens=batch_tokens[0],
                masked_tokens=masked_tokens[0],
                alphabet=alphabet
            )
        
        return masked_tokens, mlm_labels
    
    except Exception as e:
        logger.error(f"Error in collate_batch: {e}")
        logger.error(f"Number of sequences: {len(sequences)}")
        logger.error(f"First sequence: {sequences[0][:50]}...")
        
        # Print detailed information about each sequence in the batch
        for i, (seq_id, seq) in enumerate(zip(seq_ids, sequences)):
            logger.error(f"Sequence {i}: ID={seq_id}, Length={len(seq)}")
            logger.error(f"  Preview: {seq[:30]}...")
            
            # Check for unusual characters
            unusual_chars = set(seq) - set("ACDEFGHIKLMNPQRSTVWY")
            if unusual_chars:
                logger.error(f"  Contains unusual characters: {unusual_chars}")
        
        raise

def freeze_layers(model, num_frozen_layers):
    """
    Freeze the first num_frozen_layers of the transformer model
    
    Args:
        model: ESM model
        num_frozen_layers: Number of layers to freeze
    """
    total_layers = len(model.layers)
    num_frozen_layers = min(num_frozen_layers, total_layers)
    
    logger.info(f"Model has {total_layers} transformer layers total")
    logger.info(f"Will freeze {num_frozen_layers} layers, keeping {total_layers - num_frozen_layers} layers trainable")
    
    # Freeze embedding layer
    logger.info("Freezing embedding layer")
    for param in model.embed_tokens.parameters():
        param.requires_grad = False
    
    # Freeze specified transformer layers
    for i in range(num_frozen_layers):
        logger.info(f"Freezing layer {i}")
        for param in model.layers[i].parameters():
            param.requires_grad = False
    
    # Log parameter status
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"Frozen {num_frozen_layers} out of {total_layers} transformer layers")
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    # Print which layers are frozen vs trainable for clarity
    if logger.level <= logging.DEBUG:
        for name, param in model.named_parameters():
            status = "Trainable" if param.requires_grad else "Frozen"
            logger.debug(f"{status}: {name}")

class TrainingPlotter:
    """
    Class for real-time plotting of training metrics
    """
    def __init__(self, output_dir):
        """
        Initialize the plotter with empty data and create the output directory
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.epochs = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
        # Create figures
        self.loss_fig, self.loss_ax = plt.subplots(figsize=(10, 6))
        self.acc_fig, self.acc_ax = plt.subplots(figsize=(10, 6))
        self.lr_fig, self.lr_ax = plt.subplots(figsize=(10, 6))
        
        # Set up the plots
        self._setup_plots()
    
    def _setup_plots(self):
        """Set up the initial plot configurations"""
        # Loss plot
        self.loss_ax.set_title('Training and Validation Loss')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True)
        
        # Accuracy plot
        self.acc_ax.set_title('Training and Validation Accuracy')
        self.acc_ax.set_xlabel('Epoch')
        self.acc_ax.set_ylabel('Accuracy')
        self.acc_ax.grid(True)
        
        # Learning rate plot
        self.lr_ax.set_title('Learning Rate')
        self.lr_ax.set_xlabel('Epoch')
        self.lr_ax.set_ylabel('Learning Rate')
        self.lr_ax.grid(True)
        self.lr_ax.set_yscale('log')
        self.lr_ax.yaxis.set_major_formatter(ScalarFormatter())
    
    def update(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, learning_rate=None):
        """
        Update the plots with new data
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
            learning_rate: Current learning rate (optional)
        """
        # Update data
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        # Update loss plot
        self.loss_ax.clear()
        self.loss_ax.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        if len(self.val_losses) > 0:
            self.loss_ax.plot(self.epochs[:len(self.val_losses)], self.val_losses, 'r-', label='Validation Loss')
        self.loss_ax.set_title('Training and Validation Loss')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.legend()
        self.loss_ax.grid(True)
        
        # Update accuracy plot
        self.acc_ax.clear()
        self.acc_ax.plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        if len(self.val_accuracies) > 0:
            self.acc_ax.plot(self.epochs[:len(self.val_accuracies)], self.val_accuracies, 'r-', label='Validation Accuracy')
        self.acc_ax.set_title('Training and Validation Accuracy')
        self.acc_ax.set_xlabel('Epoch')
        self.acc_ax.set_ylabel('Accuracy')
        self.acc_ax.legend()
        self.acc_ax.grid(True)
        
        # Update learning rate plot
        if len(self.learning_rates) > 0:
            self.lr_ax.clear()
            self.lr_ax.plot(self.epochs[:len(self.learning_rates)], self.learning_rates, 'g-')
            self.lr_ax.set_title('Learning Rate')
            self.lr_ax.set_xlabel('Epoch')
            self.lr_ax.set_ylabel('Learning Rate')
            self.lr_ax.grid(True)
            self.lr_ax.set_yscale('log')
            self.lr_ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Save plots
        self.loss_fig.savefig(os.path.join(self.output_dir, 'loss_plot.png'))
        self.acc_fig.savefig(os.path.join(self.output_dir, 'accuracy_plot.png'))
        if len(self.learning_rates) > 0:
            self.lr_fig.savefig(os.path.join(self.output_dir, 'learning_rate_plot.png'))
        
        # Close the plots to prevent memory leaks
        plt.close(self.loss_fig)
        plt.close(self.acc_fig)
        plt.close(self.lr_fig)
        
        # Recreate the figures for the next update
        self.loss_fig, self.loss_ax = plt.subplots(figsize=(10, 6))
        self.acc_fig, self.acc_ax = plt.subplots(figsize=(10, 6))
        self.lr_fig, self.lr_ax = plt.subplots(figsize=(10, 6))
        
        # Set up the plots again
        self._setup_plots()

def fine_tune(
    model, 
    alphabet, 
    train_dataset, 
    output_dir, 
    val_dataset=None,
    batch_size=1,
    num_epochs=5,
    learning_rate=1e-5,
    mask_prob=0.15,
    num_frozen_layers=30,  # Freeze most of the layers by default (ESM-1b has 33 layers)
    save_checkpoint_steps=1000,
    scheduler_type='step',
    lr_step_size=1,
    lr_gamma=0.9,
    lr_patience=1
):
    """
    Fine-tune an ESM model on a protein MSA dataset
    
    Args:
        model: ESM model
        alphabet: ESM alphabet
        train_dataset: Training dataset
        output_dir: Directory to save model checkpoints
        val_dataset: Validation dataset (optional)
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        mask_prob: Probability of masking a token for MLM
        num_frozen_layers: Number of layers to freeze (from the beginning)
        save_checkpoint_steps: Save a checkpoint every N steps
        scheduler_type: Type of learning rate scheduler ('step', 'plateau', 'cosine', 'exponential', 'constant')
        lr_step_size: Step size for StepLR scheduler (epochs)
        lr_gamma: Gamma for StepLR and ExponentialLR schedulers
        lr_patience: Patience for ReduceLROnPlateau scheduler (epochs)
    
    Returns:
        Fine-tuned model
    """
    logger.info(f"Fine-tuning ESM model for {num_epochs} epochs with learning rate {learning_rate}")
    logger.info(f"Freezing first {num_frozen_layers} layers")
    logger.info(f"Using learning rate scheduler: {scheduler_type}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
    
    # Set up plotter for real-time visualization
    plots_dir = os.path.join(output_dir, "plots")
    plotter = TrainingPlotter(plots_dir)
    logger.info(f"Real-time plots will be saved to {plots_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Freeze specified layers
    if num_frozen_layers > 0:
        freeze_layers(model, num_frozen_layers)
    
    # Set up data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, alphabet, mask_prob),
        num_workers=0  # Adjust based on your system
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, alphabet, mask_prob),
            num_workers=0
        )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Set up scheduler based on type
    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
        logger.info(f"Using StepLR scheduler with step_size={lr_step_size}, gamma={lr_gamma}")
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_gamma, patience=lr_patience, verbose=True
        )
        logger.info(f"Using ReduceLROnPlateau scheduler with patience={lr_patience}, factor={lr_gamma}")
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        logger.info(f"Using CosineAnnealingLR scheduler with T_max={num_epochs}")
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_gamma
        )
        logger.info(f"Using ExponentialLR scheduler with gamma={lr_gamma}")
    elif scheduler_type == 'constant':
        scheduler = None
        logger.info("Using constant learning rate (no scheduler)")
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, defaulting to StepLR")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
    
    # Set up loss function
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Debug: log model architecture summary
    if logger.level <= logging.DEBUG:
        try:
            logger.debug("\n=== MODEL ARCHITECTURE ===")
            logger.debug(f"Model type: {type(model).__name__}")
            logger.debug(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            logger.debug(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            # Log layer information
            for name, module in model.named_children():
                logger.debug(f"Layer: {name}, Type: {type(module).__name__}, "
                           f"Params: {sum(p.numel() for p in module.parameters()):,}, "
                           f"Trainable: {sum(p.numel() for p in module.parameters() if p.requires_grad):,}")
        except Exception as e:
            logger.error(f"Error logging model architecture: {e}")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # Log per-batch details including tokens and predictions
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            # Get input tokens and labels
            tokens = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Log batch shapes and content for debugging
            if batch_idx == 0 or (batch_idx % 10 == 0 and logger.level <= logging.DEBUG):
                logger.debug(f"\n=== BATCH {batch_idx} DETAILS ===")
                logger.debug(f"Tokens shape: {tokens.shape}")
                logger.debug(f"Labels shape: {labels.shape}")
                
                # Count mask tokens
                mask_count = (tokens == alphabet.mask_idx).sum().item()
                total_tokens = tokens.numel()
                logger.debug(f"Mask tokens: {mask_count}/{total_tokens} ({mask_count/total_tokens*100:.2f}%)")
                
                # Show token distribution
                token_values, token_counts = torch.unique(tokens, return_counts=True)
                token_dist = {val.item(): count.item() for val, count in zip(token_values, token_counts)}
                logger.debug(f"Token distribution: {token_dist}")
                
                # Show labels distribution
                label_values, label_counts = torch.unique(labels, return_counts=True)
                label_dist = {val.item(): count.item() for val, count in zip(label_values, label_counts)}
                logger.debug(f"Labels distribution: {label_dist}")
                
                # Count non-negative labels (tokens being predicted)
                non_neg_labels = (labels >= 0).sum().item()
                logger.debug(f"Tokens being predicted: {non_neg_labels}/{labels.numel()} ({non_neg_labels/labels.numel()*100:.2f}%)")
            
            # Forward pass
            forward_start_time = time.time()
            results = model(tokens, repr_layers=[])
            logits = results["logits"]
            forward_time = time.time() - forward_start_time
            
            # Calculate loss
            loss_start_time = time.time()
            loss = loss_fn(logits.view(-1, len(alphabet)), labels.view(-1))
            loss_time = time.time() - loss_start_time
            
            # Debug: log predictions vs actual for the first few tokens of first sequence
            if batch_idx == 0 and logger.level <= logging.DEBUG:
                try:
                    # Get the first sequence
                    first_seq_tokens = tokens[0]
                    first_seq_labels = labels[0]
                    first_seq_logits = logits[0]
                    
                    # Find masked positions where we're making predictions
                    masked_positions = torch.where(first_seq_labels >= 0)[0]
                    if len(masked_positions) > 0:
                        # Take the first few masked positions
                        preview_positions = masked_positions[:min(5, len(masked_positions))]
                        
                        logger.debug("\n--- PREDICTION PREVIEW ---")
                        for pos in preview_positions:
                            # Get the correct label amino acid
                            label_idx = first_seq_labels[pos].item()
                            label_aa = None
                            for aa, idx in alphabet.tok_to_idx.items():
                                if idx == label_idx:
                                    label_aa = aa
                                    break
                            
                            # Get the predicted amino acid
                            pred_idx = first_seq_logits[pos].argmax().item()
                            pred_aa = None
                            for aa, idx in alphabet.tok_to_idx.items():
                                if idx == pred_idx:
                                    pred_aa = aa
                                    break
                            
                            # Calculate probability of correct prediction
                            probs = torch.softmax(first_seq_logits[pos], dim=0)
                            correct_prob = probs[label_idx].item()
                            
                            logger.debug(f"Position {pos}: True={label_aa}({label_idx}), "
                                       f"Pred={pred_aa}({pred_idx}), "
                                       f"Confidence={correct_prob:.4f}")
                except Exception as e:
                    logger.error(f"Error logging predictions: {e}")
            
            # Backward pass and optimization
            backward_start_time = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - backward_start_time
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate accuracy on masked tokens
            with torch.no_grad():
                for i in range(tokens.size(0)):
                    masked_indices = torch.where(labels[i] != -100)[0]
                    if len(masked_indices) > 0:
                        target_tokens = labels[i][masked_indices]
                        predicted_tokens = logits[i, masked_indices].argmax(dim=1)
                        correct = (predicted_tokens == target_tokens).sum().item()
                        epoch_correct += correct
                        epoch_total += len(masked_indices)
            
            # Log progress with more detail
            if batch_idx % 10 == 0:
                # Calculate batch accuracy
                batch_acc = 0
                if epoch_total > 0:
                    batch_acc = epoch_correct / epoch_total
                
                logger.info(f"Batch {batch_idx}: loss={loss.item():.4f}, acc={batch_acc:.4f}, "
                           f"lr={optimizer.param_groups[0]['lr']:.2e}, "
                           f"forward_time={forward_time:.4f}s, "
                           f"loss_time={loss_time:.4f}s, "
                           f"backward_time={backward_time:.4f}s")
                
                # Log to TensorBoard
                writer.add_scalar('train/batch_loss', loss.item(), global_step)
                writer.add_scalar('train/batch_accuracy', batch_acc, global_step)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            
            # Save checkpoint
            global_step += 1
            if global_step % save_checkpoint_steps == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if scheduler is not None else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'loss': loss.item(),
                    'global_step': global_step
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = epoch_correct / max(1, epoch_total)  # Avoid division by zero
        
        logger.info(f"Epoch {epoch+1} completed: avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")
        
        # Log to TensorBoard
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/epoch_accuracy', accuracy, epoch)
        
        # Validation
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            logger.info("Starting validation")
            val_start_time = time.time()
            model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                    # Get masked tokens and move to device
                    tokens = batch[0].to(device)
                    labels = batch[1].to(device)
                    
                    # Forward pass
                    results = model(tokens, repr_layers=[])
                    logits = results["logits"]
                    
                    # Calculate loss
                    loss = loss_fn(logits.view(-1, len(alphabet)), labels.view(-1))
                    val_losses.append(loss.item())
                    
                    # Calculate accuracy
                    for i in range(tokens.size(0)):
                        masked_indices = torch.where(labels[i] != -100)[0]
                        if len(masked_indices) > 0:
                            target_tokens = labels[i][masked_indices]
                            predicted_tokens = logits[i, masked_indices].argmax(dim=1)
                            correct = (predicted_tokens == target_tokens).sum().item()
                            val_correct += correct
                            val_total += len(masked_indices)
                    
                    if batch_idx == 0 or batch_idx % 50 == 0:
                        logger.info(f"Validation batch {batch_idx}: loss={loss.item():.4f}")
            
            # Calculate and log validation metrics
            val_time = time.time() - val_start_time
            val_loss = np.mean(val_losses)
            val_accuracy = val_correct / max(1, val_total)
            logger.info(f"Validation completed in {val_time:.2f}s - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
            
            # Log to TensorBoard
            writer.add_scalar('validation/loss', val_loss, epoch)
            writer.add_scalar('validation/accuracy', val_accuracy, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(output_dir, "best_model.pt")
                logger.info(f"New best validation loss: {val_loss:.4f}, saving model to {best_model_path}")
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, best_model_path)
                logger.info(f"Best model saved successfully")
        
        # Update learning rate scheduler
        if scheduler is not None:
            if scheduler_type == 'plateau':
                # ReduceLROnPlateau needs validation loss
                if val_loss is not None:
                    prev_lr = optimizer.param_groups[0]['lr']
                    scheduler.step(val_loss)
                    new_lr = optimizer.param_groups[0]['lr']
                    if new_lr != prev_lr:
                        logger.info(f"Learning rate adjusted: {prev_lr:.2e} -> {new_lr:.2e}")
                else:
                    logger.warning("ReduceLROnPlateau scheduler requires validation data, but none provided. Skipping scheduler step.")
            else:
                # Other schedulers just need to be stepped each epoch
                prev_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != prev_lr:
                    logger.info(f"Learning rate adjusted: {prev_lr:.2e} -> {new_lr:.2e}")
        
        # Update plots after each epoch
        plotter.update(
            epoch=epoch + 1,  # 1-indexed for display
            train_loss=avg_loss,
            train_acc=accuracy,
            val_loss=val_loss,
            val_acc=val_accuracy,
            learning_rate=optimizer.param_groups[0]['lr']
        )
        logger.info(f"Updated training plots in {plots_dir}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "esm1b_finetuned.pt")
    logger.info(f"Training completed. Saving final model to {final_model_path}")
    torch.save({
        'epoch': num_epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
    }, final_model_path)
    logger.info(f"Final model saved successfully")
    
    # Close tensorboard writer
    writer.close()
    logger.info("TensorBoard writer closed")
    
    # Log final training stats
    logger.info(f"Fine-tuning completed: {num_epochs} epochs, {global_step} steps")
    if val_dataset is not None:
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return model

def main():
    """Main function to run the fine-tuning process"""
    start_time = time.time()
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Fine-tune ESM-1b on MSA data")
        parser.add_argument("--msa-file", type=str, required=True, help="Path to MSA file (FASTA or A3M format)")
        parser.add_argument("--output-dir", type=str, required=True, help="Directory to save model checkpoints")
        parser.add_argument("--max-seqs", type=int, default=1000, help="Maximum number of sequences to use from MSA")
        parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length to include")
        parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
        parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
        parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
        parser.add_argument("--mask-prob", type=float, default=0.15, help="Probability of masking a token for MLM")
        parser.add_argument("--frozen-layers", type=int, default=30, help="Number of layers to freeze (from beginning)")
        parser.add_argument("--sample-method", type=str, default="random", choices=["random", "first", "evenly"], 
                           help="Method to sample sequences from MSA")
        parser.add_argument("--msa-format", type=str, default="auto", choices=["auto", "fasta", "a3m"], help="MSA file format")
        parser.add_argument("--log-level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                            help="Logging level")
        parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
        parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save training plots (default: output_dir/plots)")
        parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of data to use for validation (0-1)")
        parser.add_argument("--scheduler", type=str, default="step", 
                            choices=["step", "plateau", "cosine", "exponential", "constant"], 
                            help="Learning rate scheduler type")
        parser.add_argument("--lr-step-size", type=int, default=1, help="Step size for StepLR scheduler (epochs)")
        parser.add_argument("--lr-gamma", type=float, default=0.9, help="Gamma for StepLR and ExponentialLR schedulers")
        parser.add_argument("--lr-patience", type=int, default=1, help="Patience for ReduceLROnPlateau scheduler (epochs)")
        args = parser.parse_args()
        
        # Set up logging
        numeric_level = getattr(logging, args.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {args.log_level}")
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(args.output_dir, "fine_tuning.log"), mode='w')
            ]
        )
        
        # Debug mode: Limit the number of sequences for faster debugging
        if args.log_level.upper() == "DEBUG" and args.max_seqs > 100:
            logger.warning(f"Debug mode: Limiting sequences to 100 instead of {args.max_seqs}")
            args.max_seqs = 100
            args.batch_size = min(args.batch_size, 2)
            logger.warning(f"Debug mode: Setting batch size to {args.batch_size}")
            
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Log script parameters
        logger.info(f"Starting ESM-1b fine-tuning with parameters:")
        logger.info(f"  MSA file: {args.msa_file}")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Max sequences: {args.max_seqs}")
        logger.info(f"  Max sequence length: {args.max_length}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info(f"  Mask probability: {args.mask_prob}")
        logger.info(f"  Frozen layers: {args.frozen_layers}")
        logger.info(f"  Sample method: {args.sample_method}")
        logger.info(f"  MSA format: {args.msa_format}")
        logger.info(f"  Log level: {args.log_level}")
        logger.info(f"  Random seed: {args.seed}")
        logger.info(f"  Validation split: {args.validation_split}")
        logger.info(f"  Scheduler type: {args.scheduler}")
        
        # Set random seeds for reproducibility
        logger.info(f"Setting random seed to {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            logger.info("Also set CUDA random seed")
        
        # Load ESM-1b model and alphabet
        logger.info("Loading ESM-1b model")
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        
        # Log alphabet information for debugging
        logger.info(f"Alphabet size: {len(alphabet)}")
        logger.info(f"Special tokens: CLS={alphabet.cls_idx}, EOS={alphabet.eos_idx}, MASK={alphabet.mask_idx}, PAD={alphabet.padding_idx}")
        
        # Print a few token mappings for debugging
        try:
            logger.info(f"Token mappings (first 10): {list(alphabet.tok_to_idx.items())[:10]}")
            
            # Print all valid amino acid tokens
            valid_aas = "ACDEFGHIKLMNPQRSTVWY"
            aa_indices = {aa: alphabet.get_idx(aa) for aa in valid_aas}
            logger.info(f"Amino acid token indices: {aa_indices}")
            
            # Check if gap character has a token
            try:
                gap_idx = alphabet.get_idx('-')
                logger.info(f"Gap character '-' has token index: {gap_idx}")
            except KeyError:
                logger.info("Gap character '-' does not have a token index, will be removed from sequences")
                
        except Exception as e:
            logger.warning(f"Could not print all alphabet token mappings: {e}")
        
        # Create dataset
        logger.info(f"Creating dataset from MSA file: {args.msa_file}")
        full_dataset = ProteinMSADataset(
            msa_file=args.msa_file,
            alphabet=alphabet,
            max_seqs=args.max_seqs,
            max_length=args.max_length,
            sample_method=args.sample_method,
            msa_format=args.msa_format
        )
        logger.info(f"Dataset created with {len(full_dataset)} sequences")
        
        # Split dataset into training and validation
        if args.validation_split > 0:
            # Calculate split sizes
            val_size = int(len(full_dataset) * args.validation_split)
            train_size = len(full_dataset) - val_size
            
            # Always include the first sequence (wild-type) in training
            train_indices = [0]  # Start with the first sequence
            remaining_indices = list(range(1, len(full_dataset)))
            random.shuffle(remaining_indices)
            
            # Add remaining indices to reach train_size
            train_indices.extend(remaining_indices[:train_size-1])
            val_indices = remaining_indices[train_size-1:]
            
            # Create subset datasets
            from torch.utils.data import Subset
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            
            logger.info(f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation sequences")
        else:
            train_dataset = full_dataset
            val_dataset = None
            logger.info("Using all sequences for training (no validation split)")
        
        # Fine-tune model
        logger.info("Starting fine-tuning process")
        fine_tuned_model = fine_tune(
            model=model,
            alphabet=alphabet,
            train_dataset=train_dataset,
            output_dir=args.output_dir,
            val_dataset=val_dataset,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            mask_prob=args.mask_prob,
            num_frozen_layers=args.frozen_layers,
            scheduler_type=args.scheduler,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            lr_patience=args.lr_patience
        )
        
        # Log completion
        end_time = time.time()
        logger.info(f"Fine-tuning completed in {end_time - start_time:.2f} seconds")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1) 