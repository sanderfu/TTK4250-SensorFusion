    @singledispatchmethod
    def init_filter_state(self, init) -> None:
        raise NotImplementedError(
            f"EKF do not know how to make {init} into GaussParams"
        )

    @init_filter_state.register(GaussParams)
    def _(self, init: GaussParams) -> GaussParams:
        return init

    @init_filter_state.register(tuple)
    @init_filter_state.register(list)
    def _(self, init: Union[Tuple, List]) -> GaussParams:
        return GaussParams(*init)

    @init_filter_state.register(dict)
    def _(self, init: dict) -> GaussParams:
        got_mean = False
        got_cov = False

        for key in init:
            if not got_mean and key in ["mean", "x", "m"]:
                mean = init[key]
                got_mean = True
            if not got_cov and key in ["cov", "P"]:
                cov = init[key]
                got_cov = True

        assert (
            got_mean and got_cov
        ), f"EKF do not recognize mean and cov keys in the dict {init}."

        return GaussParams(mean, cov)
