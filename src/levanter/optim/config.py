import abc
import re
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Optional

import draccus
import equinox as eqx
import jax
import numpy as np
import optax
from jax import numpy as jnp

import haliax

import levanter.tracker
from levanter.utils.jax_utils import leaf_key_paths


@dataclass
class OptimizerConfig(draccus.ChoiceRegistry, abc.ABC):
    learning_rate: float = 6e-4
    weight_decay: float = 0.1

    min_lr_ratio: float = 0.1
    """The lr scheduler operates on 4 stages: [warmup] - {[stable] - [decay]} x haps - [cooldown]"""
    warmup: int | float = 0.01
    """fraction of training steps to use as warmup, or steps to use. 0.0 means no warmup"""
    decay: int | float | None = None
    """fraction of training steps to use as decay, or steps to use. None means full decay"""
    rewarmup: int | float = 0.0
    "If using a cycle, how much of the cycle to use as re-warmup. 0.0 means no re-warmup."
    cooldown: Optional[float] = None
    """Deprecated, as its semantics are confusing."""
    cycle_length: int | float | None | list[int] = None
    """ Length of cycle. If <= 1, it is treated as a fraction of the total number of steps. None is equivalent to 1.0."""
    cycles: int | list[int] | None = None
    """Number of cycles or a list of cycle endpoints. Can use at most one of cycle_length, cycles, or haps."""

    lr_schedule: str = "cosine"  # constant, cosine, linear
    stable_lr_schedule: str = "constant"
    haps: Optional[list[int]] = None
    """Deprecated."""
    weight_decay_modules: Optional[list[str] | str] = None
    """A regex or a list of strings to identify where to mask weight.
    For nano-GPT, this field can be set as `r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings"`"""
    default_weight_decay_mask: Optional[bool] = None
    """Whether to apply a default reasonable weight decay to modules not explicitly masked. None means it will if
    no weight_decay_modules are set. False means it will not. True means it will regardless of weight_decay_modules."""

    @classmethod
    def default_choice_name(cls) -> Optional[str]:
        return "adam"

    @abc.abstractmethod
    def build(self, num_train_steps: int):
        raise NotImplementedError

    def build_weight_decay_mask(self):
        def reasonable_default(module, path):
            # TODO: gross
            if "LayerNorm" in path:
                return False
            if "RMSNorm" in path:
                return False
            if "RmsNorm" in path:
                return False
            if "Embedding" in path:
                return False
            if path.endswith("bias"):
                return False
            return None

        if self.weight_decay_modules is None and self.default_weight_decay_mask is False:
            return None
        else:
            should_use_default = self.default_weight_decay_mask is True or (
                self.default_weight_decay_mask is None and self.weight_decay_modules is None
            )

            def is_leaf(x):
                return eqx.is_array(x) or isinstance(x, eqx.Module) or haliax.is_named_array(x)

            # mask based on regex or module path
            def _apply_on(decayed_paths, x, from_root_key_path, from_class_keypath):
                if isinstance(x, eqx.Module):
                    is_leaf_here = lambda y: x is not y and is_leaf(y)  # noqa: E731
                    # we want to support both Linear.weight and transformer.encoder.layers.0.mlp.dense.weight
                    class_name = x.__class__.__name__
                    # recursively apply to submodules.
                    from_root_key_paths = leaf_key_paths(x, is_leaf=is_leaf_here, prefix=from_root_key_path)
                    from_class_key_paths = leaf_key_paths(x, is_leaf=is_leaf_here, prefix=class_name)
                    this_mask = jax.tree_util.tree_map(
                        partial(_apply_on, decayed_paths),
                        x,
                        from_root_key_paths,
                        from_class_key_paths,
                        is_leaf=lambda y: x is not y and is_leaf(y),
                    )
                    return this_mask
                elif not haliax.util.is_jax_or_hax_array_like(x):
                    return x

                should_decay = None
                for key_path in [from_root_key_path, from_class_keypath]:
                    if key_path is None:
                        continue

                    if isinstance(self.weight_decay_modules, str):
                        compiled_regex = re.compile(self.weight_decay_modules)
                        should_decay = should_decay or compiled_regex.match(key_path) is not None
                    elif isinstance(self.weight_decay_modules, list):
                        should_decay = should_decay or any(
                            key_path.__contains__(target) for target in self.weight_decay_modules
                        )

                    if should_use_default and not should_decay:
                        should_decay = reasonable_default(x, key_path)

                    if should_decay:
                        break

                if should_decay is None:
                    if should_use_default:
                        should_decay = True
                    else:
                        should_decay = False

                if should_decay:
                    decayed_paths.append(from_root_key_path)

                return should_decay

            def mask_fn(model):
                decayed_paths = []
                mask = jax.tree_util.tree_map(
                    partial(_apply_on, decayed_paths, from_class_keypath=None),
                    model,
                    leaf_key_paths(model, is_leaf=is_leaf),
                    is_leaf=is_leaf,
                )

                # log all decayed weights
                levanter.tracker.log_hyperparameters({"decayed_weights": sorted(decayed_paths)})

                return mask

            return mask_fn

    def lr_scheduler(self, num_train_steps):
        if self.cooldown is not None:
            warnings.warn("cooldown is deprecated. Just use the normal schedule.", DeprecationWarning)
            cooldown_steps = _convert_frac_or_steps(self.cooldown, num_train_steps)
        else:
            cooldown_steps = 0

        total_main_steps = num_train_steps - cooldown_steps
        cooldown_points = self._get_cycle_minima(total_main_steps)

        min_lr = self.learning_rate * self.min_lr_ratio

        schedules = []
        boundaries = []

        previous_end = 0.0

        for cycle, (start, end) in enumerate(zip(cooldown_points[:-1], cooldown_points[1:])):
            cycle_steps = end - start
            if cycle == 0:  # warmup
                warmup_steps = _convert_frac_or_steps(self.warmup, cycle_steps)
            else:
                warmup_steps = _convert_frac_or_steps(self.rewarmup, cycle_steps)

            if warmup_steps != 0:
                warmup = optax.linear_schedule(previous_end, self.learning_rate, warmup_steps)
                schedules.append(warmup)
                boundaries.append(start + warmup_steps)

            lr_decay_steps = (
                _convert_frac_or_steps(self.decay, cycle_steps)
                if self.decay is not None
                else cycle_steps - warmup_steps
            )
            stable_steps = cycle_steps - warmup_steps - lr_decay_steps
            
            
            final_stable_lr = self.learning_rate
            if stable_steps != 0:
                match self.stable_lr_schedule:
                    case "constant":
                        stable = optax.constant_schedule(self.learning_rate)
                    case "cosine":
                        stable = optax.cosine_decay_schedule(self.learning_rate, lr_decay_steps, self.min_lr_ratio)
                    case "linear":
                        stable = optax.linear_schedule(self.learning_rate, min_lr, lr_decay_steps)
                    case "inv_sqrt":
                        stable = _inv_sqrt_decay_schedule(self.learning_rate, min_lr, warmup_steps, 10000)
                    case "inv":
                        stable = _inv_decay_schedule(self.learning_rate, min_lr, lr_decay_steps)
                    case "fitted":
                        stable = _fitted_lr_schedule(self.learning_rate, min_lr, lr_decay_steps)
                    case _:
                        raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
                schedules.append(stable)
                boundaries.append(start + warmup_steps + stable_steps)
                final_stable_lr = stable(stable_steps)
                
                
            match self.lr_schedule:
                case "constant":
                    schedule = optax.constant_schedule(final_stable_lr)
                case "cosine":
                    schedule = optax.cosine_decay_schedule(final_stable_lr, lr_decay_steps, self.min_lr_ratio)
                case "linear":
                    schedule = optax.linear_schedule(final_stable_lr, min_lr, lr_decay_steps)
                case "inv_sqrt":
                    schedule = _inv_sqrt_decay_schedule(final_stable_lr, min_lr, warmup_steps, 10000)
                case "inv":
                    schedule = _inv_decay_schedule(final_stable_lr, min_lr, lr_decay_steps)
                case "fitted":
                    schedule = _fitted_lr_schedule(final_stable_lr, min_lr, lr_decay_steps)
                case _:
                    raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

            previous_end = schedule(lr_decay_steps)

            schedules.append(schedule)
            boundaries.append(end)

        if cooldown_steps != 0:
            final_main_lr = schedule(lr_decay_steps)
            cooldown = optax.linear_schedule(final_main_lr, min_lr, cooldown_steps)
            schedules.append(cooldown)

        if len(schedules) > 1:
            schedule = optax.join_schedules(schedules, boundaries)
        else:
            schedule = schedules[0]

        return schedule

    def _get_cycle_minima(self, total_main_steps):
        if self.cycle_length is not None:
            if self.cycles is not None:
                raise ValueError("Can't use both cycle_length and cycles.")
            if self.haps is not None:
                warnings.warn("haps is deprecated. Use cycles instead.", DeprecationWarning)
                raise ValueError("Can't use both cycle_length and haps.")

            if isinstance(self.cycle_length, int | float):
                cycle_length = _convert_frac_or_steps(self.cycle_length, total_main_steps)
                cooldown_points = [i * cycle_length for i in range(1, total_main_steps // cycle_length)]
                if total_main_steps % cycle_length != 0:
                    warnings.warn(
                        "Cycle length does not divide total number of steps. The last cycle will be shorter."
                    )

            elif isinstance(self.cycle_length, list):
                lengths = np.array(self.cycle_length)
                steps = np.cumsum(lengths)
                if steps[-1] > total_main_steps:
                    raise ValueError(f"Cycle lengths exceed total number of steps: {steps[-1]} > {total_main_steps}")
                cooldown_points = steps.tolist()
            else:
                raise ValueError("Invalid cycle_length. Must be a fraction, number of steps, or a list of steps.")

        elif self.haps is not None:
            warnings.warn("haps is deprecated. Use cycles instead.", DeprecationWarning)
            cooldown_points = list(self.haps)
        elif isinstance(self.cycles, int):
            # insert a warmup then the rest of the steps
            cooldown_points = [int(total_main_steps / self.cycles * (i + 1)) for i in range(self.cycles - 1)]
        elif isinstance(self.cycles, list):
            cooldown_points = list(self.cycles)
        else:
            cooldown_points = []

        cooldown_points.insert(0, 0)
        if cooldown_points[-1] != total_main_steps:
            cooldown_points.append(total_main_steps)
        return cooldown_points


def _inv_sqrt_decay_schedule(lr: float, min_lr: float, warmup_steps: int, timescale: float = 10000):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / jnp.sqrt(jnp.maximum(count + warmup_steps, 1) / timescale))
        return jnp.maximum(lr * decay, min_lr)

    return schedule


def _inv_decay_schedule(lr: float, min_lr: float, decay_steps: int):
    def schedule(count):
        decay = jnp.minimum(1.0, 1.0 / ((lr / min_lr - 1) * jnp.maximum(count, 1) / decay_steps + 1))
        return jnp.maximum(lr * decay, min_lr)

    return schedule

def _fitted_lr_schedule(lr: float, min_lr:float, decay_steps: int):
    sampled_lrs = jnp.array([0.0007999999797903001, 0.0007994223851710558, 0.0007988402503542602, 0.0007982535171322525, 0.0007976623019203544, 0.0007970664883032441, 0.0007964661344885826, 0.0007958612404763699, 0.0007952517480589449, 0.0007946377154439688, 0.0007940191426314414, 0.0007933960296213627, 0.0007927683764137328, 0.0007921361830085516, 0.0007914993911981583, 0.0007908580591902137, 0.0007902121869847178, 0.0007895617163740098, 0.0007889067637734115, 0.000788247212767601, 0.0007875830633565784, 0.0007869143737480044, 0.0007862411439418793, 0.0007855633157305419, 0.0007848809473216534, 0.0007841939805075526, 0.0007835024734959006, 0.0007828064262866974, 0.000782105780672282, 0.0007814005366526544, 0.0007806907524354756, 0.0007799764280207455, 0.0007792575052008033, 0.0007785340421833098, 0.0007778059807606041, 0.0007770733791403472, 0.0007763361791148782, 0.0007755944388918579, 0.0007748481584712863, 0.0007740972796455026, 0.0007733418024145067, 0.0007725817849859595, 0.0007718172273598611, 0.0007710480713285506, 0.0007702743750996888, 0.0007694960804656148, 0.0007687132456339896, 0.0007679258706048131, 0.0007671338971704245, 0.0007663373253308237, 0.0007655362132936716, 0.0007647305610589683, 0.0007639203104190528, 0.0007631055195815861, 0.0007622861303389072, 0.0007614622008986771, 0.0007606337312608957, 0.0007598006632179022, 0.0007589629967696965, 0.0007581207901239395, 0.0007572740432806313, 0.0007564226980321109, 0.0007555668125860393, 0.0007547063287347555, 0.0007538413046859205, 0.0007529716822318733, 0.0007520975195802748, 0.0007512187585234642, 0.0007503355154767632, 0.0007494476158171892, 0.0007485551759600639, 0.0007476581959053874, 0.0007467566174454987, 0.0007458504987880588, 0.0007449397817254066, 0.0007440245244652033, 0.0007431046687997878, 0.000742180272936821, 0.000741251278668642, 0.0007403176859952509, 0.0007393794949166477, 0.0007384367636404932, 0.0007374894339591265, 0.0007365375058725476, 0.0007355810375884175, 0.0007346199708990753, 0.0007336543058045208, 0.0007326841005124152, 0.0007317092968150973, 0.0007307298947125673, 0.0007297458942048252, 0.0007287573534995317, 0.0007277642143890262, 0.0007267665350809693, 0.0007257641991600394, 0.0007247573812492192, 0.0007237459067255259, 0.0007227298920042813, 0.0007217092788778245, 0.0007206840673461556, 0.0007196543156169355, 0.0007186199654825032, 0.0007175810169428587, 0.000716537469998002, 0.0007154893828555942, 0.000714436755515635, 0.0007133794715628028, 0.0007123176474124193, 0.0007112512248568237, 0.0007101802038960159, 0.0007091046427376568, 0.0007080244831740856, 0.0007069397252053022, 0.0007058504270389676, 0.0007047565304674208, 0.0007036580936983228, 0.0007025550003163517, 0.0007014473667368293, 0.0007003351347520947, 0.000699218362569809, 0.000698096991982311, 0.0006969710229896009, 0.0006958404555916786, 0.0006947053479962051, 0.0006935656419955194, 0.0006924213957972825, 0.0006912725511938334, 0.0006901191081851721, 0.0006889610667712986, 0.000687798485159874, 0.0006866313051432371, 0.0006854595267213881, 0.0006842832081019878, 0.0006831022910773754, 0.0006819167756475508, 0.000680726720020175, 0.000679532065987587, 0.0006783328135497868, 0.0006771289627067745, 0.0006759205716662109, 0.0006747075822204351, 0.0006734899943694472, 0.0006722677499055862, 0.000671040965244174, 0.0006698096403852105, 0.000668573658913374, 0.0006673330790363252, 0.0006660879007540643, 0.0006648381822742522, 0.000663583807181567, 0.0006623248918913305, 0.0006610613781958818, 0.0006597932078875601, 0.0006585204973816872, 0.000657243188470602, 0.0006559612811543047, 0.0006546747754327953, 0.0006533836713060737, 0.0006520879687741399, 0.0006507877260446548, 0.0006494828267022967, 0.0006481733871623874, 0.0006468593492172658, 0.0006455406546592712, 0.0006442174199037254, 0.0006428895867429674, 0.0006415571551769972, 0.0006402201252058148, 0.0006388785550370812, 0.0006375323282554746, 0.0006361815612763166, 0.0006348261376842856, 0.0006334661738947034, 0.0006321015534922481, 0.0006307323928922415, 0.0006293586338870227, 0.0006279802764765918, 0.0006265973206609488, 0.0006252097664400935, 0.000623817672021687, 0.0006224209209904075, 0.0006210196297615767, 0.0006196136819198728, 0.0006182031938806176, 0.0006167880492284894, 0.0006153683643788099, 0.0006139441393315792, 0.0006125152576714754, 0.0006110817193984985, 0.0006096436991356313, 0.000608201022259891, 0.0006067538051865995, 0.0006053019315004349, 0.000603845517616719, 0.000602384505327791, 0.0006009188946336508, 0.0005994486855342984, 0.0005979738780297339, 0.0005964944721199572, 0.0005950104678049684, 0.0005935219232924283, 0.0005920287221670151, 0.0005905309808440506, 0.0005890285829082131, 0.0005875217029824853, 0.0005860101664438844, 0.0005844939732924104, 0.0005829732399433851, 0.0005814479663968086, 0.000579918036237359, 0.0005783835076726973, 0.0005768443807028234, 0.0005753006553277373, 0.0005737522733397782, 0.0005721993511542678, 0.0005706417723558843, 0.0005690795951522887, 0.0005675128195434809, 0.0005659414455294609, 0.0005643654731102288, 0.0005627848440781236, 0.0005611996748484671, 0.0005596098490059376, 0.0005580154829658568, 0.0005564164603129029, 0.0005548128392547369, 0.0005532046779990196, 0.0005515918019227684, 0.0005499743856489658, 0.0005483523709699512, 0.0005467256996780634, 0.0005450944881886244, 0.0005434586200863123, 0.000541818211786449, 0.0005401731468737125, 0.000538523425348103, 0.0005368691636249423, 0.0005352103617042303, 0.0005335468449629843, 0.0005318787880241871, 0.0005302060744725168, 0.0005285288207232952, 0.0005268469103612006, 0.0005251604015938938, 0.0005234692944213748, 0.0005217735888436437, 0.0005200732848607004, 0.0005183683824725449, 0.0005166588816791773, 0.0005149447824805975, 0.0005132260266691446, 0.0005115026724524796, 0.0005097747780382633, 0.000508042168803513, 0.0005063050193712115, 0.0005045633297413588, 0.0005028169834986329, 0.000501065980643034, 0.0004993104375898838, 0.0004975502961315215, 0.000495785498060286, 0.0004940161015838385, 0.0004922421649098396, 0.0004904635716229677, 0.0004886804381385446, 0.00048689261893741786, 0.00048510023043490946, 0.0004833032435271889, 0.00048150165821425617, 0.0004796954453922808, 0.0004778846341650933, 0.0004760692536365241, 0.00047424924559891224, 0.0004724246100522578, 0.0004705954052042216, 0.00046876160195097327, 0.0004669231711886823, 0.0004650801420211792, 0.0004632325144484639, 0.00046138028847053647, 0.00045952346408739686, 0.0004576620412990451, 0.00045579602010548115, 0.0004539253714028746, 0.0004520501533988863, 0.00045017030788585544, 0.0004482858639676124, 0.00044639682164415717, 0.00044450315181165934, 0.0004426048544701189, 0.00044070195872336626, 0.000438794435467571, 0.0004368823138065636, 0.0004349655646365136, 0.00043304418795742095, 0.0004311182419769466, 0.00042918763938359916, 0.00042725243838503957, 0.00042531260987743735, 0.000423368182964623, 0.00042141915764659643, 0.00041946550481952727, 0.0004175072244834155, 0.00041554434574209154, 0.00041357683949172497, 0.00041160473483614624, 0.0004096280026715249, 0.00040764667210169137, 0.00040566071402281523, 0.0004036701575387269, 0.000401674973545596, 0.0003996751911472529, 0.00039767081034369767, 0.0003956618020310998, 0.0003936481662094593, 0.00039162993198260665, 0.00038960709935054183, 0.0003875796392094344, 0.00038554755155928433, 0.00038351089460775256, 0.0003814696101471782, 0.00037942369817756116, 0.000377373187802732, 0.00037531807902269065, 0.0003732583427336067, 0.0003711940080393106, 0.0003691250749398023, 0.0003670515143312514, 0.00036497332621365786, 0.0003628905687946826, 0.00036080318386666477, 0.00035871120053343475, 0.0003566145896911621, 0.0003545133804436773, 0.00035240757279098034, 0.00035029713762924075, 0.000348182104062289, 0.0003460624720901251, 0.00034393821260891855, 0.00034180935472249985, 0.000339675898430869, 0.00033753784373402596, 0.0003353951615281403, 0.0003332478809170425, 0.0003310960019007325, 0.0003289395244792104, 0.0003267784195486456, 0.0003246127162128687, 0.0003224424144718796, 0.00032026751432567835, 0.00031808801577426493, 0.00031590391881763935, 0.00031371519435197115, 0.0003115218714810908, 0.00030932395020499825, 0.00030712143052369356, 0.0003049143124371767, 0.0003027025959454477, 0.0003004862810485065, 0.00029826536774635315, 0.0002960397978313267, 0.00029380968771874905, 0.0002915749792009592, 0.0002893356722779572, 0.0002870917087420821, 0.0002848432050086558, 0.0002825901028700173, 0.00028033240232616663, 0.0002780701033771038, 0.0002758032060228288, 0.00027353165205568075, 0.00027125555789098144, 0.00026897486532106996, 0.0002666895743459463, 0.0002643996849656105, 0.0002621051389724016, 0.0002598060527816415, 0.00025750230997800827, 0.0002551939687691629, 0.00025288102915510535, 0.00025056349113583565, 0.0002482413547113538, 0.00024591461988165975, 0.00024358322843909264, 0.00024124729679897428, 0.00023890676675364375, 0.00023656158009544015, 0.0002342118532396853, 0.00023185752797871828, 0.0002294985461048782, 0.00022713502403348684, 0.00022476690355688334, 0.00022239418467506766, 0.00022001686738803983, 0.00021763495169579983, 0.00021524843759834766, 0.00021285732509568334, 0.00021046167239546776, 0.00020806142129004002, 0.0002056565135717392, 0.00020324712386354804, 0.0002008330775424838, 0.0001984144328162074, 0.00019599124789237976, 0.00019356352277100086, 0.00019113114103674889, 0.00018869421910494566, 0.00018625269876793027, 0.00018380663823336363, 0.00018135597929358482, 0.00017890078015625477, 0.00017644098261371255, 0.00017397658666595817, 0.00017150770872831345, 0.00016903417417779565, 0.00016655615763738751, 0.00016407354269176722, 0.00016158632934093475, 0.00015909463400021195, 0.000156598340254277, 0.00015409750631079078, 0.0001515921321697533, 0.0001490822178311646, 0.00014656770508736372, 0.0001440487103536725, 0.00014152517542243004, 0.00013899710029363632, 0.00013646448496729136, 0.00013392732944339514, 0.00013138569192960858, 0.00012883951421827078, 0.00012628885451704264, 0.00012373365461826324, 0.00012117397272959352, 0.00011860980885103345, 0.00011604110477492213, 0.00011346797691658139, 0.00011089036706835032, 0.0001083082752302289, 0.00010572170140221715, 0.00010313064558431506, 0.00010053522419184446, 9.793532080948353e-05, 9.533099364489317e-05, 9.272224269807339e-05, 9.01091261766851e-05, 8.749158587306738e-05, 8.486973820254207e-05, 8.224346674978733e-05, 7.961282972246408e-05, 7.697794353589416e-05, 7.433869177475572e-05, 7.16951908543706e-05, 6.904744077473879e-05, 6.63954415358603e-05, 6.373925134539604e-05, 6.107887020334601e-05, 5.8414414525032043e-05, 5.574582610279322e-05, 5.307322135195136e-05, 5.0396600272506475e-05, 4.77161374874413e-05, 4.5031774789094925e-05, 4.2343686800450087e-05, 3.96519317291677e-05, 3.695662599056959e-05, 3.4258002415299416e-05, 3.1556177418679e-05, 2.885144203901291e-05, 2.6144087314605713e-05, 2.343446249142289e-05, 2.072314964607358e-05, 1.8010905478149652e-05, 1.529889414086938e-05, 1.258880365639925e-05, 9.883602615445852e-06, 7.187190931290388e-06, 4.502420779317617e-06, 1.6396516002714634e-06, 1.6122357919812202e-06])
    step = jnp.arange(0, 47840, 1)
    sampled_points = jnp.arange(0, 47840, 100)
    sampled_points = jnp.concatenate([sampled_points, jnp.array([47839])]) 
    optimal_lr = jnp.interp(step, sampled_points, sampled_lrs)
    def schedule(count):
        return optimal_lr[count]
    return schedule


def _convert_frac_or_steps(frac_or_steps: float | int, num_train_steps: int):
    # if it's greater than 1, it must be a whole number of steps
    if frac_or_steps < 0.0 or (frac_or_steps > 1.0 and frac_or_steps % 1 != 0):
        raise ValueError(f"Invalid fraction {frac_or_steps}. Must be between 0 and 1. You can also use (whole) steps.")
    if frac_or_steps <= 1.0:
        return int(frac_or_steps * num_train_steps)

    return int(frac_or_steps)


@dataclass
class HessianOptConfig(OptimizerConfig, abc.ABC):
    update_interval: int = 10
    """How often to update the hessian approximation."""


@OptimizerConfig.register_subclass("adam")
@dataclass
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    # cf https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.DecoupledAdamW.html
    # https://x.com/giffmana/status/1692641748445438301
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    nesterov: bool = False

    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon, nesterov = self.nesterov))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


@OptimizerConfig.register_subclass("lion")
@dataclass
class LionConfig(OptimizerConfig):
    beta1: float = 0.9
    # cf https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.DecoupledAdamW.html
    # https://x.com/giffmana/status/1692641748445438301
    beta2: float = 0.95
    max_grad_norm: Optional[float] = 1.0

    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_lion(self.beta1, self.beta2))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))
