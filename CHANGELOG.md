# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/PythonPredictions/cobra/compare/v1.1.0...HEAD)</small>

### Added

- Addapt requirements to scikit-learn>=1.2.0 and Change tests for newer sklearn ([2d610d7](https://github.com/PythonPredictions/cobra/commit/2d610d7af407d07ef057024b7c4efc4900106e63) by Patrick Leonardy).
- Add: automatic seach for cont/discont variables ([bf3967e](https://github.com/PythonPredictions/cobra/commit/bf3967ebfb37782a18004902a4f545f69374e835) by Patrick Leonardy).
- Add new line at end of file, Format with black ([8820580](https://github.com/PythonPredictions/cobra/commit/88205803549a72d288c106570a3646efff455e45) by Patrick Leonardy).
- added type hint for test_col_id ([a9c21ca](https://github.com/PythonPredictions/cobra/commit/a9c21caa12c7a5afda01f22b1c16ac2d3e30a927) by Patrick Leonardy).
- added example 4 categorical in test_drops_columns_containing_only_nan ([d598002](https://github.com/PythonPredictions/cobra/commit/d598002c9bed1d430cb99a539c8ee7158d9d2b1c) by Patrick Leonardy).
- Added drop of columns containing only NANs ([2ac2a3d](https://github.com/PythonPredictions/cobra/commit/2ac2a3d507963292a00bfdb8a615cb3074c300de) by Patrick Leonardy).

### Fixed

- fix: dev: #154 Invoke: fixed nb task for Windows systems ([50ba7e2](https://github.com/PythonPredictions/cobra/commit/50ba7e2c47f36d526403acd4b908e712e483b2e9) by Pietro D'Antuono).
- fix: dev: Minor fix #154 ([ef891b3](https://github.com/PythonPredictions/cobra/commit/ef891b3b02747a1317080e51622d87d60f29ba53) by Pietro D'Antuono).
- fix: dev: Fixed inv test command on Windows machines  #154. ([7aecf53](https://github.com/PythonPredictions/cobra/commit/7aecf53fc6344dc43b8d090c55f08510e634a6f8) by Pietro D'Antuono).
- fix: dev: Fixed inv test command on Windows machines. ([5e4109a](https://github.com/PythonPredictions/cobra/commit/5e4109aa16a18132454e3b115d180b47450b9d63) by Pietro D'Antuono).
- Fix typos in preprocessor ([e838cea](https://github.com/PythonPredictions/cobra/commit/e838cea251178da89ea72b9e9366ccc83ad7714e) by Patrick Leonardy).
- fix: mutable objects copied locally ([3759dc8](https://github.com/PythonPredictions/cobra/commit/3759dc82a54c203a647151ea609dbe2d5a1c0587) by ZlaTanskY).
- fix failing tests ([83b194b](https://github.com/PythonPredictions/cobra/commit/83b194b0cf5dc14d6339bf9a2befd24b6aa310ef) by sborms).

### Changed

- change method name in test ([c1e2725](https://github.com/PythonPredictions/cobra/commit/c1e2725b3ce040e95d5d6a8150389e0d5cc1e2ba) by Patrick Leonardy).
- Change to the behavior if no id_col_name is given ([58afbfa](https://github.com/PythonPredictions/cobra/commit/58afbfabc4fb7ecb413c1ef340dc4599e8ea9a37) by Patrick Leonardy).

<!-- insertion marker -->
## [v1.1.0](https://github.com/PythonPredictions/cobra/releases/tag/v1.1.0) - 2021-10-05

<small>[Compare with v1.0.2](https://github.com/PythonPredictions/cobra/compare/v1.0.2...v1.1.0)</small>

### Added

- add comma ([14e5bf2](https://github.com/PythonPredictions/cobra/commit/14e5bf2c8b808ce059743f783fd001b3683b73ef) by sborms).
- add __version__ string ([51f2285](https://github.com/PythonPredictions/cobra/commit/51f228532f5d76b37b1b779e0eba61550af8351c) by sborms).
- additional explanations in docs ([183535c](https://github.com/PythonPredictions/cobra/commit/183535c4d769ee99723db8d846ad7a6600d65b77) by sborms).
- add cleaned unrun tutorials under new folder ([9e12942](https://github.com/PythonPredictions/cobra/commit/9e12942d25dc3a48008bda6d36dbf0b9bc528eca) by sborms).
- add self.model_type ([beaf97b](https://github.com/PythonPredictions/cobra/commit/beaf97b9206b4718320f12f04f96fdea0d3a7816) by sborms).
- add model_type functionality in PreProcessor class & tests ([07338fb](https://github.com/PythonPredictions/cobra/commit/07338fb213dd269abe104c12a6887c4dac0b6fb1) by sborms).
- added model_type parameter, different statistical test for regression models, included a simple test ([7f339fd](https://github.com/PythonPredictions/cobra/commit/7f339fde8590b0c9b7af6d282450b49a0914a1dd) by sborms).
- added pip build (#78) ([47c23b5](https://github.com/PythonPredictions/cobra/commit/47c23b5ba4db0dffbd0657519634adde7669a15e) by Jan Benisek).

### Fixed

- fix train-selection split ffs ([ffd852e](https://github.com/PythonPredictions/cobra/commit/ffd852e7dce70a472a17e283e4dadc6c8dfabf34) by sborms).
- fix raise ValueError optimal step ([834c869](https://github.com/PythonPredictions/cobra/commit/834c869a9024088aee1279c8d691bc214e85165a) by sborms).
- fix README ([c073418](https://github.com/PythonPredictions/cobra/commit/c0734186f4a809853bd008bb10bde30d36f0c7c0) by sborms).
- Fixed the showing of a warning with the formatting of ticks (which apparently is a known bug in the latest version) by ignoring warnings from matplotlib. I tried different things but this is the only thing that worked. ([137c00a](https://github.com/PythonPredictions/cobra/commit/137c00a2bca8629a040ff49505926413fb3b2f83) by hendrik.dewinter).
- fixed the legend warning in qqplot by adding labels to the plot elements ([a66c0ce](https://github.com/PythonPredictions/cobra/commit/a66c0ce3632879ba6b0f80ae546f2cc67b94cf25) by hendrik.dewinter).
- fix qq plot ([09e51c6](https://github.com/PythonPredictions/cobra/commit/09e51c601885133a52e28b4e297df4aa9b26cd3c) by sborms).
- Fix: model scoring did not work in the regression case (indexing error - linear regression model does not have TWO columns). ([ac522ea](https://github.com/PythonPredictions/cobra/commit/ac522eaaf81b7ff4cb8e2a4e101fff476e0a083b) by Sander Vanden Hautte).

### Changed

- change link logo ([a2bb0c1](https://github.com/PythonPredictions/cobra/commit/a2bb0c191191fe4af97801871621fa27e9a53879) by sborms).

### Removed

- removed junit (#72) ([ae6a746](https://github.com/PythonPredictions/cobra/commit/ae6a7460fc50b3429c6b57c2c911dbc2c0afd404) by Jan Benisek).

## [v1.0.2](https://github.com/PythonPredictions/cobra/releases/tag/v1.0.2) - 2021-07-12

<small>[Compare with v1.0.1](https://github.com/PythonPredictions/cobra/compare/v1.0.1...v1.0.2)</small>

### Added

- Add logo & badges (#59) ([6ebe0a0](https://github.com/PythonPredictions/cobra/commit/6ebe0a081f8ecc1acbbe62f19ce816ca0fb3f83e) by Sam Borms).
- added status badge ([94020b3](https://github.com/PythonPredictions/cobra/commit/94020b3644a7884c4e7406b1b5471bbb96691ae7) by Jan Benisek).
- added back pylint ([22af742](https://github.com/PythonPredictions/cobra/commit/22af742a3f5d32413a08f0b96aae27580472dd12) by Jan Benisek).
- added CI action ([24a7293](https://github.com/PythonPredictions/cobra/commit/24a7293333812085e96759d434cf523e8dd41756) by JanBenisek).
- added link to wiki and move contribute guidelines ([1d550cd](https://github.com/PythonPredictions/cobra/commit/1d550cde2b304d2af1d89601ec355cc9f52dfaeb) by Jan Benisek).
- added .github with templates ([f72b9a5](https://github.com/PythonPredictions/cobra/commit/f72b9a563639f67f34d81ff5072c36245201ab8f) by JanBenisek).

### Fixed

- fix #37 set copy warning ([b1ccf74](https://github.com/PythonPredictions/cobra/commit/b1ccf74fef7fdcef3ca383bc1937f624797894bc) by Jan Beníšek).
- Fixes #39 - added warning ([6764bc6](https://github.com/PythonPredictions/cobra/commit/6764bc6140c4f1b59760404756f01e36bb98b87a) by JanBenisek).
- fix CI ([3550e16](https://github.com/PythonPredictions/cobra/commit/3550e167ff422a733639f2c09a1eec08ed15ac14) by Jan Benisek).
- fixes #39 by using inf ([b66a0c4](https://github.com/PythonPredictions/cobra/commit/b66a0c4a90a727ffc9041047fd8ca190ccc1a62a) by JanBenisek).
- fixed #40 - improved code, added unit test, removed mistake ([c12792b](https://github.com/PythonPredictions/cobra/commit/c12792b51d9aacf4f61c35119f916d41729edcc5) by JanBenisek).
- fixed #40 - improved code, added unit test ([428a37d](https://github.com/PythonPredictions/cobra/commit/428a37db36813a77c5155ad2e71d2657fe88bafb) by JanBenisek).
- fixed #40 by ignoring constants ([df87aca](https://github.com/PythonPredictions/cobra/commit/df87aca138881c151f44b877cc1d2528c67597e0) by JanBenisek).
- fixed link to wiki ([58d2e2b](https://github.com/PythonPredictions/cobra/commit/58d2e2bad39f8cba607dab1587ced4df5a090c62) by Jan Benisek).

### Removed

- remove .idea + add to .gitignore ([17c8c60](https://github.com/PythonPredictions/cobra/commit/17c8c6060ef15e91f21f1dd6c68c33004a1185d5) by sborms).
- removed pylint ([b80b1d3](https://github.com/PythonPredictions/cobra/commit/b80b1d3ecbb0325f3d89955b87d810b3f348cca3) by Jan Benisek).

## [v1.0.1](https://github.com/PythonPredictions/cobra/releases/tag/v1.0.1) - 2020-12-22

<small>[Compare with v1.0.0](https://github.com/PythonPredictions/cobra/compare/v1.0.0...v1.0.1)</small>

### Added

- Added code for plotting PIGs (#23) ([002bc57](https://github.com/PythonPredictions/cobra/commit/002bc572190176e552349cbbf8a9dc927553235d) by Jan Benisek).
- Add LICENSE ([6afe41b](https://github.com/PythonPredictions/cobra/commit/6afe41b1c6c24f6676a2657e8ec8f98a8a042c5e) by Matthias).

## [v1.0.0](https://github.com/PythonPredictions/cobra/releases/tag/v1.0.0) - 2020-06-26

<small>[Compare with v0.0.1](https://github.com/PythonPredictions/cobra/compare/v0.0.1...v1.0.0)</small>

### Added

- Add matthews correlation coeff as evaluation metric ([3865b36](https://github.com/PythonPredictions/cobra/commit/3865b36bbf9961fdc5b8682e93aa4b5c48fa3b9a) by Matthias Roels).
- Add additional unittest for TargetEncoder ([f2c71ed](https://github.com/PythonPredictions/cobra/commit/f2c71edc9922da31f96e18f7b55919e8c6d346a8) by Matthias Roels).
- Add option to save figure to plotting_utils ([581f7b4](https://github.com/PythonPredictions/cobra/commit/581f7b4daf753051465700c78bc4aca7410d4cd5) by Matthias Roels).
- Add documentation to evaluation module ([63b584a](https://github.com/PythonPredictions/cobra/commit/63b584ad0b5258348b12e68ef7e361b30bc20648) by Matthias Roels).
- Add cumulative gains metric to Evaluator ([ec1903a](https://github.com/PythonPredictions/cobra/commit/ec1903a60d15ea2e68d0e3ac4415dfcdb7282eb8) by Matthias Roels).
- Add Evaluator class ([0d4b157](https://github.com/PythonPredictions/cobra/commit/0d4b1571e4837b061542795043cf2861786490ca) by Matthias Roels).
- Add evaluation to setup.py ([2c47dd2](https://github.com/PythonPredictions/cobra/commit/2c47dd215168738a1b96c63a90300f215b899e8c) by Matthias Roels).
- Add plotting functions to evaluation module ([baf792f](https://github.com/PythonPredictions/cobra/commit/baf792f438ad4fc251897fa405f6b2cfeb84726a) by Matthias Roels).
- Add evaluation module with PIGs script ([234aba8](https://github.com/PythonPredictions/cobra/commit/234aba8551c03e05450a24ea8a50d5b56339e71f) by Matthias Roels).
- Add default args to univariate_selection ([3b65235](https://github.com/PythonPredictions/cobra/commit/3b6523585e8a690675e39cc3925dae8e7c400321) by Matthias Roels).
- Add docstrings to forward_selection.py ([e94f17d](https://github.com/PythonPredictions/cobra/commit/e94f17d809afb20ca49b9e4727de6b12db1e4768) by Matthias Roels).
- Add usage section to README ([fd2d3bc](https://github.com/PythonPredictions/cobra/commit/fd2d3bccbc97435136b83f9f71c44e942fce818e) by Matthias Roels).
- Add variable importance computation to models.py ([aaff470](https://github.com/PythonPredictions/cobra/commit/aaff470f24fc16b854b75d5659366e04012f409b) by Matthias Roels).
- Add functions to explore result of forward_selection ([f05f154](https://github.com/PythonPredictions/cobra/commit/f05f15479941b4f881d5b6fda81bbeb474b8b102) by Matthias Roels).
- Add convenience function to univariate selection ([4a5195c](https://github.com/PythonPredictions/cobra/commit/4a5195cc6028ab6a52f023fe34fa4e12358f6165) by Matthias Roels).
- Add forward_selection submodule to model_building module ([dc2b1ea](https://github.com/PythonPredictions/cobra/commit/dc2b1eaefd2f5f97629c1dd445e422ac2211051c) by Matthias Roels).
- Add model_building module ([ea4f3ef](https://github.com/PythonPredictions/cobra/commit/ea4f3eff4044acb1d9fb18c1cc917f2c3f571c0d) by Matthias Roels).
- Add PreProcessor class as facade for preprocessing ([5e4a06d](https://github.com/PythonPredictions/cobra/commit/5e4a06dbccbdf89253666469d20a64c977daf577) by Matthias Roels).
- Add fit_transform method to TargetEncoder ([cff611e](https://github.com/PythonPredictions/cobra/commit/cff611e74b4a761a3ebfb38fcf56b4e54d3c1ad7) by Matthias Roels).
- Add (de)serializers to CategoricalDataProcessor ([75eb75c](https://github.com/PythonPredictions/cobra/commit/75eb75c3a8d64cf935ddd9445f57e24320010b36) by Matthias Roels).
- Add additional log statement to target_encoder ([49b9227](https://github.com/PythonPredictions/cobra/commit/49b922781b4d26c444a5a8e086b4d6e5a8b70a0d) by Matthias Roels).
- Added forced category option to CategoricalDataProcessor ([77a4336](https://github.com/PythonPredictions/cobra/commit/77a43363e96e1aeca92462cb3a8fce24d6fb677a) by Matthias Roels).
- Add more unittests for KBinsDiscretizer ([447e7fa](https://github.com/PythonPredictions/cobra/commit/447e7fae280f2b97d7a1f3f51d2d338b1a034ede) by Matthias Roels).
- Add (de)serialization to KBinsDiscretizer ([4ffc7f0](https://github.com/PythonPredictions/cobra/commit/4ffc7f07b04bfaf55612e7813c0c6feaf9ea1018) by Matthias Roels).
- Add categorical regrouper ([20dc24b](https://github.com/PythonPredictions/cobra/commit/20dc24b66b37fca56c50ea33c4df693dedb0cf26) by JanBenisek).
- Add more docstrings to preprocessing module methods ([e596c12](https://github.com/PythonPredictions/cobra/commit/e596c12154aba6e10b806cc306a106bc01a9a712) by Matthias Roels).
- Add model building module ([67703d3](https://github.com/PythonPredictions/cobra/commit/67703d3d7cd6f310b1ca3acc83732217123e1efc) by Matthias Roels).
- Add matrics module to cobra ([dde17de](https://github.com/PythonPredictions/cobra/commit/dde17de1990fbdfd45da19d9ba49f06ef7ceb9a1) by Matthias Roels).
- added  cobra_env ([72adaa3](https://github.com/PythonPredictions/cobra/commit/72adaa31f9e7760eae8fd1a2132ac232f2584532) by JanBenisek).
- Add unittests for all private methods of KBinsDiscretizer ([7f1a2b8](https://github.com/PythonPredictions/cobra/commit/7f1a2b893420046d94460870331544cbdebd646f) by Matthias Roels).
- Add a scripts module with a first example script ([0b89146](https://github.com/PythonPredictions/cobra/commit/0b89146e4811bd737e05f113ed9eb3563920dfa3) by Matthias Roels).
- Add TargetEncoder to preprocessing module ([41a79cb](https://github.com/PythonPredictions/cobra/commit/41a79cb308a1df46238b8bb8b596b009ac02178d) by Matthias Roels).
- Add KBinsDiscretizer to new preprocessing module ([b109798](https://github.com/PythonPredictions/cobra/commit/b109798f737a59991b7d6b9ae2ea5166438e1a68) by Matthias Roels).

### Fixed

- Fix bug in KBinsDiscretizer.set_attributes_from_dict ([8d8d553](https://github.com/PythonPredictions/cobra/commit/8d8d5534b04f0adfc9b199d06146b1c17af3b0e6) by Matthias Roels).
- Fix random state in models, bug fixing in evaluator ([9a5299a](https://github.com/PythonPredictions/cobra/commit/9a5299ac771ede3fc48d8c67110fd0b5c10abe46) by Matthias Roels).
- Fix typo in README ([69041d3](https://github.com/PythonPredictions/cobra/commit/69041d3da31b426cd2fa24b030587b69dd662720) by Matthias Roels).
- Fix a bug in univariate_selection.compute_univariate_preselection output ([b6dac15](https://github.com/PythonPredictions/cobra/commit/b6dac15cec4f810eb934ba38aef232b67ab1c222) by Matthias Roels).
- Fix a bug in KBinsDiscretizer.fit method ([31165d7](https://github.com/PythonPredictions/cobra/commit/31165d71dfa153c75b53e3397efe5c5cab067d5d) by Matthias Roels).
- Fix in unittest for CategoricalDataProcessor ([08eda64](https://github.com/PythonPredictions/cobra/commit/08eda64beefffd1a457ade8966687f215d03a5c3) by Matthias Roels).

### Changed

- Change datatype of Evaluator.scalar_metrics to pd.Series ([a68ef38](https://github.com/PythonPredictions/cobra/commit/a68ef3814a4c01c879ac6771e3f3dbd241494bc3) by Matthias Roels).
- Change color scheme in Evaluator plots ([748f681](https://github.com/PythonPredictions/cobra/commit/748f6819ce237e93b92ea26592bb07f4ddb8b8e0) by Matthias Roels).
- Change return type of Models.compute_variable_importance ([21cc7a7](https://github.com/PythonPredictions/cobra/commit/21cc7a735517bd549bb22fae276edab18bc73db7) by Matthias Roels).
- Change line endings to linux style in evaluation module ([746c99b](https://github.com/PythonPredictions/cobra/commit/746c99b7fac3528b1d1adf4932f020248395d7b6) by Matthias Roels).
- Change output of forward_selection.compute_model_performances ([e2635ae](https://github.com/PythonPredictions/cobra/commit/e2635aea2ff4d706d5344efb9bae81e4878918f8) by Matthias Roels).
- Change docstring format in univariate_selection.py ([946a500](https://github.com/PythonPredictions/cobra/commit/946a500718050ccefc77c242f7643fb17b641ae8) by Matthias Roels).
- Change API of PreProcessor for extra flexibility ([3e88237](https://github.com/PythonPredictions/cobra/commit/3e8823775a7ef678809a78474934015af388eca8) by Matthias Roels).
- Change logic of CategoricalDataProcessor ([6cc18a9](https://github.com/PythonPredictions/cobra/commit/6cc18a9d53a62e94463786d64d990724ad8f7817) by Matthias Roels).

### Removed

- Remove old refactored modules and extend utils.py ([8d87d32](https://github.com/PythonPredictions/cobra/commit/8d87d32f668c3c15064207ed519d9a0a0b3c1dd9) by Matthias Roels).
- Remove legacy code from repo as part of clean-up ([2872707](https://github.com/PythonPredictions/cobra/commit/287270711a285ee1fd2647b992f97ad661d044d5) by Matthias Roels).
- Remove metrics module which was put in the wrong branch ([03b5b4d](https://github.com/PythonPredictions/cobra/commit/03b5b4d3b957b0f3dd1b3f7699d9f29f497f8296) by Matthias Roels).

## [v0.0.1](https://github.com/PythonPredictions/cobra/releases/tag/v0.0.1) - 2018-10-26

<small>[Compare with first commit](https://github.com/PythonPredictions/cobra/compare/75694aeca596a4ee6cbf5e5e12259dd3e0fac9e7...v0.0.1)</small>

### Added

- added intercept and verbosity ([e013647](https://github.com/PythonPredictions/cobra/commit/e01364768548e1539b0af812b8a3eba67ccc956e) by JanBenisek).
- added verbose option ([8920b41](https://github.com/PythonPredictions/cobra/commit/8920b411223baf41f745845603faef429e4228e9) by JanBenisek).
- Added summary, modified AUC plot ([5ce5f94](https://github.com/PythonPredictions/cobra/commit/5ce5f948b49647f05d623e9b9e2a66dc546294ee) by JanBenisek).

### Fixed

- FIX: attributes availability ([6122984](https://github.com/PythonPredictions/cobra/commit/6122984e913b257854807bd3fb339c588b576ec7) by JanBenisek).
- FIX: modeling_nsteps +1 ([94aa2fd](https://github.com/PythonPredictions/cobra/commit/94aa2fdf8c3e36513b8b41863ee416e0cee67ab0) by JanBenisek).
- FIX: Multuple lines in model output ([93cacbf](https://github.com/PythonPredictions/cobra/commit/93cacbf719cf5ed34075abeeb66e0ad338b91f78) by JanBenisek).
- FIX verbose ([9b9f5e8](https://github.com/PythonPredictions/cobra/commit/9b9f5e84ab210c59a1d69554c3ced93c657c8680) by JanBenisek).
- FIX: indendation ([e7cedf5](https://github.com/PythonPredictions/cobra/commit/e7cedf59b671f5fa10e45cc55f31017d79e95e05) by JanBenisek).
- FIX: incidence plot (handle missing) ([497ea60](https://github.com/PythonPredictions/cobra/commit/497ea605031150bdaa6568d412231e13e15d3227) by JanBenisek).
- FIX: import with missings ([c9e5bd7](https://github.com/PythonPredictions/cobra/commit/c9e5bd76ea653a8b87adc6182ae3f1e0bcf423a4) by JanBenisek).
- FIX: import regex ([22ef160](https://github.com/PythonPredictions/cobra/commit/22ef16003ccbe78dc175687f5e389fd42b924a9f) by JanBenisek).
- FIX: small patches for plots ([489efc5](https://github.com/PythonPredictions/cobra/commit/489efc5505cc85691688bd32ed92146c60a2771d) by JanBenisek).
- FIX: No models with positive coefs ([4cb33b6](https://github.com/PythonPredictions/cobra/commit/4cb33b6cb1a5408b379839d7d1bf9b9c487d7b51) by JanBenisek).
- fixed FS, plots, usability ([4827c60](https://github.com/PythonPredictions/cobra/commit/4827c60bc406c9118d4a7ea6eb7a533101eed2fd) by JanBenisek).

