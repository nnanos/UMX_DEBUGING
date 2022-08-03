import argparse
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable

import torch
import torch.utils.data
import torchaudio
import tqdm
#from Audio_proc_lib.audio_proc_functions import load_music,sound_write




def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds

    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


def load_audio(
    path: str,
    start: float = 0.0,
    dur: Optional[float] = None,
    info: Optional[dict] = None,
):
    """Load audio file

    Args:
        path: Path of audio file
        start: start position in seconds, defaults on the beginning.
        dur: end position in seconds, defaults to `None` (full file).
        info: metadata object as called from `load_info`.

    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
    # loads the full track duration
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path)
        return sig, rate
    else:
        if info is None:
            info = load_info(path)
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset)
        return sig, rate


def aug_from_str(list_of_function_names: list):
    if list_of_function_names:
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def _augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    # for multichannel > 2, we drop the other channels
    if audio.shape[0] > 2:
        audio = audio[:2, ...]

    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio



class UnmixDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(
        self,
        root: Union[Path, str],
        sample_rate: float,
        seq_duration: Optional[float] = None,
        source_augmentations: Optional[Callable] = None,
    ) -> None:
        self.root = Path(args.root).expanduser()
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""



def load_datasets(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Tuple[UnmixDataset, UnmixDataset, argparse.Namespace]:
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """
    if args.dataset == "aligned":
        parser.add_argument("--input-file", type=str)
        parser.add_argument("--output-file", type=str)

        args = parser.parse_args()
        # set output target to basename of output file
        args.target = Path(args.output_file).stem

        dataset_kwargs = {
            "root": Path(args.root),
            "seq_duration": args.seq_dur,
            "input_file": args.input_file,
            "output_file": args.output_file,
        }
        args.target = Path(args.output_file).stem
        train_dataset = AlignedDataset(
            split="train", random_chunks=True, **dataset_kwargs
        )  # type: UnmixDataset
        valid_dataset = AlignedDataset(split="valid", **dataset_kwargs)  # type: UnmixDataset

    else:
        parser.add_argument(
            "--is-wav",
            action="store_true",
            default=False,
            help="loads wav instead of STEMS",
        )
        parser.add_argument("--samples-per-track", type=int, default=64)
        parser.add_argument(
            "--source-augmentations", type=str, default=["gain", "channelswap"], nargs="+"
        )

        args = parser.parse_args()
        dataset_kwargs = {
            "root": args.root,
            "is_wav": args.is_wav,
            "subsets": "train",
            "target": args.target,
            "download": args.root is None,
            "seed": args.seed,
        }

        source_augmentations = aug_from_str(args.source_augmentations)

        train_dataset = MUSDBDataset(
            split="train",
            samples_per_track=args.samples_per_track,
            seq_duration=args.seq_dur,
            source_augmentations=source_augmentations,
            random_track_mix=True,
            **dataset_kwargs,
        )

        valid_dataset = MUSDBDataset(
            split="valid", samples_per_track=1, seq_duration=None, **dataset_kwargs
        )


    return train_dataset, valid_dataset, args






class AlignedDataset(UnmixDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        input_file: str = "mixture.wav",
        output_file: str = "vocals.wav",
        seq_duration: Optional[float] = None,
        random_chunks: bool = False,
        sample_rate: float = 44100.0,
        source_augmentations: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        """A dataset of that assumes multiple track folders
        where each track includes and input and an output file
        which directly corresponds to the the input and the
        output of the model. This dataset is the most basic of
        all datasets provided here, due to the least amount of
        preprocessing, it is also the fastest option, however,
        it lacks any kind of source augmentations or custum mixing.

        Typical use cases:

        * Source Separation (Mixture -> Target)
        * Denoising (Noisy -> Clean)
        * Bandwidth Extension (Low Bandwidth -> High Bandwidth)

        Example
        =======
        data/train/01/mixture.wav --> input
        data/train/01/vocals.wav ---> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_chunks = random_chunks
        # set the input and output files (accept glob)
        self.input_file = input_file
        self.output_file = output_file
        self.tuple_paths = list(self._get_paths())
        if not self.tuple_paths:
            raise RuntimeError("Dataset is empty, please check parameters")
        self.seed = seed
        random.seed(self.seed)

    def __getitem__(self, index):
        input_path, output_path = self.tuple_paths[index]

        if self.random_chunks:
            input_info = load_info(input_path)
            output_info = load_info(output_path)
            duration = min(input_info["duration"], output_info["duration"])
            start = random.uniform(0, duration - self.seq_duration)
        else:
            start = 0

        X_audio, _ = load_audio(input_path, start=start, dur=self.seq_duration)
        Y_audio, _ = load_audio(output_path, start=start, dur=self.seq_duration)
        # return torch tensors
        return X_audio, Y_audio

    def __len__(self):
        return len(self.tuple_paths)

    def _get_paths(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                input_path = list(track_path.glob(self.input_file))
                output_path = list(track_path.glob(self.output_file))
                if input_path and output_path:
                    if self.seq_duration is not None:
                        input_info = load_info(input_path[0])
                        output_info = load_info(output_path[0])
                        min_duration = min(input_info["duration"], output_info["duration"])
                        # check if both targets are available in the subfolder
                        if min_duration > self.seq_duration:
                            yield input_path[0], output_path[0]
                    else:
                        yield input_path[0], output_path[0]






class MUSDBDataset(UnmixDataset):
    def __init__(
        self,
        target: str = "vocals",
        root: str = None,
        download: bool = False,
        is_wav: bool = False,
        subsets: str = "train",
        split: str = "train",
        seq_duration: Optional[float] = 6.0,
        samples_per_track: int = 64,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        random_track_mix: bool = False,
        seed: int = 42,
        *args,
        **kwargs,
    ) -> None:
        """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
        using track and excerpts with replacement.

        Parameters
        ----------
        target : str
            target name of the source to be separated, defaults to ``vocals``.
        root : str
            root path of MUSDB
        download : boolean
            automatically download 7s preview version of MUSDB
        is_wav : boolean
            specify if the WAV version (instead of the MP4 STEMS) are used
        subsets : list-like [str]
            subset str or list of subset. Defaults to ``train``.
        split : str
            use (stratified) track splits for validation split (``valid``),
            defaults to ``train``.
        seq_duration : float
            training is performed in chunks of ``seq_duration`` (in seconds,
            defaults to ``None`` which loads the full audio track
        samples_per_track : int
            sets the number of samples, yielded from each track per epoch.
            Defaults to 64
        source_augmentations : list[callables]
            provide list of augmentation function that take a multi-channel
            audio file of shape (src, samples) as input and output. Defaults to
            no-augmentations (input = output)
        random_track_mix : boolean
            randomly mixes sources from different tracks to assemble a
            custom mix. This augmenation is only applied for the train subset.
        seed : int
            control randomness of dataset iterations
        args, kwargs : additional keyword arguments
            used to add further control for the musdb dataset
            initialization function.

        """
        import musdb

        self.seed = seed
        random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
            *args,
            **kwargs,
        )
        self.sample_rate = 44100.0  # musdb is fixed sample rate

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == "train" and self.seq_duration:
            for k, source in enumerate(self.mus.setup["sources"]):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration

                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(0, track.duration - self.seq_duration)
                # load source audio and apply time domain source_augmentations
                audio = torch.as_tensor(track.sources[source].audio.T, dtype=torch.float32)
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup["sources"].keys()).index("vocals")
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.as_tensor(track.audio.T, dtype=torch.float32)
            y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)


        # track.chunk_duration = self.seq_duration
        # # set random start position
        # track.chunk_start = random.uniform(0, track.duration - self.seq_duration)        

        # # get the non-linear source mix straight from musdb
        # x = torch.as_tensor(track.audio.T, dtype=torch.float32)
        # y = torch.as_tensor(track.targets[self.target].audio.T, dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track




# class MUSDBDataset(UnmixDataset):
#     def __init__(
#         self,
#         target: str = "vocals",
#         root: str = None,
#         download: bool = False,
#         is_wav: bool = False,
#         subsets: str = "train",
#         split: str = "train",
#         seq_duration: Optional[float] = 6.0,
#         samples_per_track: int = 64,
#         source_augmentations: Optional[Callable] = lambda audio: audio,
#         random_track_mix: bool = False,
#         seed: int = 42,
#         *args,
#         **kwargs,
#     ) -> None:
#         """MUSDB18 torch.data.Dataset that samples from the MUSDB tracks
#         using track and excerpts with replacement.

#         Parameters
#         ----------
#         target : str
#             target name of the source to be separated, defaults to ``vocals``.
#         root : str
#             root path of MUSDB
#         download : boolean
#             automatically download 7s preview version of MUSDB
#         is_wav : boolean
#             specify if the WAV version (instead of the MP4 STEMS) are used
#         subsets : list-like [str]
#             subset str or list of subset. Defaults to ``train``.
#         split : str
#             use (stratified) track splits for validation split (``valid``),
#             defaults to ``train``.
#         seq_duration : float
#             training is performed in chunks of ``seq_duration`` (in seconds,
#             defaults to ``None`` which loads the full audio track
#         samples_per_track : int
#             sets the number of samples, yielded from each track per epoch.
#             Defaults to 64
#         source_augmentations : list[callables]
#             provide list of augmentation function that take a multi-channel
#             audio file of shape (src, samples) as input and output. Defaults to
#             no-augmentations (input = output)
#         random_track_mix : boolean
#             randomly mixes sources from different tracks to assemble a
#             custom mix. This augmenation is only applied for the train subset.
#         seed : int
#             control randomness of dataset iterations
#         args, kwargs : additional keyword arguments
#             used to add further control for the musdb dataset
#             initialization function.

#         """
#         import musdb

#         self.seed = seed
#         random.seed(seed)
#         self.is_wav = is_wav
#         self.seq_duration = seq_duration
#         self.target = target
#         self.subsets = subsets
#         self.split = split
#         self.samples_per_track = samples_per_track
#         self.source_augmentations = source_augmentations
#         self.random_track_mix = random_track_mix
#         self.mus = musdb.DB(
#             root=root,
#             is_wav=is_wav,
#             split=split,
#             subsets=subsets,
#             download=download,
#             *args,
#             **kwargs,
#         )
#         self.sample_rate = 44100.0  # musdb is fixed sample rate

#     def __getitem__(self, index):

#         track = self.mus.tracks[index // self.samples_per_track]
#         track.chunk_duration = 5.0
#         track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
#         x = track.audio.T
#         y = track.targets['vocals'].audio.T

#         return x, y

#     def __len__(self):
#         return len(self.mus.tracks) * self.samples_per_track


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')
    parser.add_argument(
        '--dataset', type=str, default="musdb",
        choices=[
            'musdb', 'aligned', 'sourcefolder',
            'trackfolder_var', 'trackfolder_fix'
        ],
        help='Name of the dataset.'
    )

    parser.add_argument(
        '--root', type=str, help='root path of dataset'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help=('write out a fixed dataset of samples')
    )

    parser.add_argument('--target', type=str, default='vocals')

    # I/O Parameters
    parser.add_argument(
        '--seq-dur', type=float, default=5.0,
        help='Duration of <=0.0 will result in the full audio'
    )

    parser.add_argument('--batch-size', type=int, default=16)

    parser.add_argument('--seed', type=int, default=42)

    args, _ = parser.parse_known_args()
    train_dataset, valid_dataset, args = load_datasets(parser, args)


    # # Iterate over training dataset
    # total_training_duration = 0
    # for k in tqdm.tqdm(range(len(train_dataset))):
    #     x, y = train_dataset[k]
    #     total_training_duration += x.shape[1] / train_dataset.sample_rate
    #     if args.save:
    #         import soundfile as sf
    #         sf.write(
    #             "test/" + str(k) + 'x.wav',
    #             x.detach().numpy().T,
    #             44100,
    #         )
    #         sf.write(
    #             "test/" + str(k) + 'y.wav',
    #             y.detach().numpy().T,
    #             44100,
    #         )

    # print("Total training duration (h): ", total_training_duration / 3600)
    # print("Number of train samples: ", len(train_dataset))
    # print("Number of validation samples: ", len(valid_dataset))

    # iterate over dataloader
    train_dataset.seq_duration = args.seq_dur
    train_dataset.random_chunks = True

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    )

    for x, y in tqdm.tqdm(train_sampler):
        pass
