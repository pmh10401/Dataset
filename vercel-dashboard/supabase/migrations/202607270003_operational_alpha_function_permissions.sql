revoke all on function public.request_operational_alpha_change(
  text, numeric, text
) from public, anon;

revoke all on function public.acknowledge_operational_alpha_change(
  text, bigint, numeric, boolean, text
) from public, anon;

grant execute on function public.request_operational_alpha_change(
  text, numeric, text
) to authenticated;

grant execute on function public.acknowledge_operational_alpha_change(
  text, bigint, numeric, boolean, text
) to authenticated;
